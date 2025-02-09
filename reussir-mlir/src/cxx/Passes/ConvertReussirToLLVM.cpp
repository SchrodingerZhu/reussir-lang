#include "Reussir/Common.h"
#include "Reussir/IR/ReussirOps.h"
#include "Reussir/IR/ReussirOpsEnums.h"
#include "Reussir/IR/ReussirTypes.h"
#include "Reussir/Interfaces/ReussirCompositeLayoutInterface.h"
#include "Reussir/Passes.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <numeric>
#include <optional>

namespace mlir {

namespace REUSSIR_DECL_SCOPE {

template <typename Op>
class TypeCoercionLowering : public mlir::OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      Op op, OpConversionPattern<Op>::OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return mlir::reussir::success();
  }
};

template <typename Op>
class ReussirConvPatternWithLayoutCache : public mlir::OpConversionPattern<Op> {
protected:
  CompositeLayoutCache &cache;

  const LLVMTypeConverter &getLLVMTypeConverter() const {
    return static_cast<const LLVMTypeConverter &>(*this->typeConverter);
  }

public:
  template <typename... Args>
  ReussirConvPatternWithLayoutCache(CompositeLayoutCache &cache, Args &&...args)
      : mlir::OpConversionPattern<Op>(std::forward<Args>(args)...),
        cache(cache) {}
};

class RcFreezeOpLowering
    : public ReussirConvPatternWithLayoutCache<RcFreezeOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;
  mlir::reussir::LogicalResult matchAndRewrite(
      RcFreezeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    if (op.getRcPtr().getType().getAtomicKind().getValue() !=
        AtomicKind::nonatomic)
      return LogicalResult::failure();
    rewriter.create<func::CallOp>(
        op->getLoc(), FlatSymbolRefAttr::get(getContext(), "__reussir_freeze"),
        TypeRange{}, adaptor.getRcPtr());
    rewriter.replaceOp(op, adaptor.getRcPtr());
    return mlir::reussir::success();
  }
};

class MRefAssignOpLowering
    : public ReussirConvPatternWithLayoutCache<MRefAssignOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;
  mlir::reussir::LogicalResult matchAndRewrite(
      MRefAssignOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        op, adaptor.getValue(), adaptor.getRefOfMRef(),
        cache.getDataLayout().getTypeABIAlignment(ptrTy));
    return mlir::reussir::success();
  }
};

class RegionCreateOpLowering
    : public ReussirConvPatternWithLayoutCache<RegionCreateOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      RegionCreateOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto ptrAlign = cache.getDataLayout().getTypeABIAlignment(ptrTy);
    auto one = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), getLLVMTypeConverter().getIndexType(), 1);
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, ptrTy, ptrTy, one,
                                                ptrAlign);
    return mlir::reussir::success();
  }
};

class DestroyOpLowering : public ReussirConvPatternWithLayoutCache<DestroyOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      DestroyOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    if (op.getObject().getType().getPointee().isIntOrIndexOrFloat())
      rewriter.eraseOp(op);
    else
      llvm_unreachable("unimplemented");
    return mlir::reussir::success();
  }
};

class UnionGetTagOpLowering
    : public ReussirConvPatternWithLayoutCache<UnionGetTagOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      UnionGetTagOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto unionTy = cast<UnionType>(op.getUnionRef().getType().getPointee());
    auto layout = cache.get(unionTy);
    auto tagTy = detail::UnionTypeImpl::getTagType(getContext(),
                                                   unionTy.getInnerTypes());
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto indexMappedTy = getLLVMTypeConverter().getIndexType();
    auto tagElementPtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), ptrTy, typeConverter->convertType(unionTy),
        adaptor.getUnionRef(), ArrayRef<LLVM::GEPArg>{0, 0});
    Value tag = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), tagTy, tagElementPtr,
        cache.getDataLayout().getTypeABIAlignment(tagTy));
    if (tag.getType() != indexMappedTy)
      tag = rewriter.create<LLVM::ZExtOp>(op->getLoc(), indexMappedTy, tag);
    rewriter.replaceOp(op, tag);
    return mlir::reussir::success();
  }
};

class UnionInspectOpLowering
    : public ReussirConvPatternWithLayoutCache<UnionInspectOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      UnionInspectOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    if (!op->getNumResults()) {
      rewriter.eraseOp(op);
      return mlir::reussir::success();
    }

    auto unionTy = cast<UnionType>(op.getUnionRef().getType().getPointee());
    const auto &layout = cache.get(unionTy);
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, LLVM::LLVMPointerType::get(getContext()),
        typeConverter->convertType(unionTy), adaptor.getUnionRef(),
        ArrayRef<LLVM::GEPArg>{0, layout.getField(1).index});
    return mlir::reussir::success();
  }
};

class CompositeAssembleOpLowering
    : public ReussirConvPatternWithLayoutCache<CompositeAssembleOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      CompositeAssembleOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto structTy = cast<CompositeType>(op.getComposite().getType());
    const auto &layout = cache.get(structTy);
    Value structVal = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), typeConverter->convertType(structTy));
    for (auto [idx, val] : llvm::enumerate(adaptor.getFields())) {
      auto field = layout.getField(idx);
      structVal = rewriter.create<LLVM::InsertValueOp>(op->getLoc(), structVal,
                                                       val, field.index);
    }
    rewriter.replaceOp(op, structVal);
    return mlir::reussir::success();
  }
};

static Value foldUnrealizedCast(Value value) {
  while (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    value = castOp.getOperand(0);
  }
  return value;
}

static void assumeAlignment(Value ptr, size_t alignment, OpBuilder &builder,
                            const LLVMTypeConverter &typeConverter) {
  auto alignmentMask = builder.create<LLVM::ConstantOp>(
      ptr.getLoc(), typeConverter.getIndexType(), alignment - 1);
  auto ptrToInt = builder.create<LLVM::PtrToIntOp>(
      ptr.getLoc(), typeConverter.getIndexType(), ptr);
  auto andOp =
      builder.create<LLVM::AndOp>(ptr.getLoc(), ptrToInt, alignmentMask);
  auto zero = builder.create<LLVM::ConstantOp>(ptr.getLoc(),
                                               alignmentMask.getType(), 0);
  auto eqOp = builder.create<LLVM::ICmpOp>(
      ptr.getLoc(), LLVM::ICmpPredicate::eq, andOp, zero);
  builder.create<LLVM::AssumeOp>(ptr.getLoc(), eqOp);
}

class RcCreateOpLowering
    : public ReussirConvPatternWithLayoutCache<RcCreateOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      RcCreateOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto rcTy = op.getType();
    unsigned dataIndex = op.getRegion() ? 3 : 1;
    auto rcBoxTy = RcBoxType::get(getContext(), rcTy.getPointee(),
                                  rcTy.getAtomicKind(), rcTy.getFreezingKind());
    const auto &layout = cache.get(rcBoxTy);
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto convertedRcBoxTy = layout.getLLVMType(getLLVMTypeConverter());
    auto dataAreaTy = convertedRcBoxTy.getTypeAtIndex(
        rewriter.getI32IntegerAttr(layout.getField(dataIndex).index));
    auto counterPtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), ptrTy, convertedRcBoxTy, adaptor.getToken(),
        ArrayRef<LLVM::GEPArg>{0, 0});
    auto valuePtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), ptrTy, convertedRcBoxTy, adaptor.getToken(),
        ArrayRef<LLVM::GEPArg>{0, layout.getField(dataIndex).index});
    auto alignment = layout.getField(dataIndex).alignment;
    assumeAlignment(valuePtr, alignment.value(), rewriter,
                    getLLVMTypeConverter());
    auto counterTy = getLLVMTypeConverter().getIndexType();
    auto one = rewriter.create<LLVM::ConstantOp>(op->getLoc(), counterTy, 1);
    rewriter.create<LLVM::StoreOp>(
        op->getLoc(), one, counterPtr,
        cache.getDataLayout().getTypeABIAlignment(counterTy));
    if (op.getRegion()) {
      auto nextPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), ptrTy, convertedRcBoxTy, adaptor.getToken(),
          ArrayRef<LLVM::GEPArg>{0, layout.getField(1).index});
      auto vtablePtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), ptrTy, convertedRcBoxTy, adaptor.getToken(),
          ArrayRef<LLVM::GEPArg>{0, layout.getField(2).index});
      // store vtable ptr to vtable field
      std::string mangledName;
      llvm::raw_string_ostream os(mangledName);
      formatMangledNameTo(rcTy.getPointee(), os);
      os << "::$fvtable";
      auto vtable = rewriter.create<LLVM::AddressOfOp>(
          op->getLoc(), ptrTy,
          FlatSymbolRefAttr::get(getContext(), mangledName));
      rewriter.create<LLVM::StoreOp>(
          op->getLoc(), vtable, vtablePtr,
          cache.getDataLayout().getTypeABIAlignment(ptrTy));
      // load tail from region ctx
      auto tail = rewriter.create<LLVM::LoadOp>(
          op->getLoc(), ptrTy, adaptor.getRegion(),
          cache.getDataLayout().getTypeABIAlignment(ptrTy));
      // store tail to next field
      rewriter.create<LLVM::StoreOp>(
          op->getLoc(), tail, nextPtr,
          cache.getDataLayout().getTypeABIAlignment(ptrTy));
      // store ourself to region ctx
      rewriter.create<LLVM::StoreOp>(
          op->getLoc(), adaptor.getToken(), adaptor.getRegion(),
          cache.getDataLayout().getTypeABIAlignment(ptrTy));
    }
    // if we know previous value is loaded, we can optimize it to use inline
    // memcpy
    if (auto load = dyn_cast_or_null<LLVM::LoadOp>(
            foldUnrealizedCast(adaptor.getValue()).getDefiningOp())) {
      rewriter.create<LLVM::MemcpyInlineOp>(
          op->getLoc(), valuePtr, load.getAddr(),
          rewriter.getI64IntegerAttr(
              cache.getDataLayout().getTypeSize(dataAreaTy)),
          false);
    } else {
      rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.getValue(), valuePtr,
                                     cache.getDataLayout().getTypeABIAlignment(
                                         adaptor.getValue().getType()));
    }
    rewriter.replaceOp(op, adaptor.getToken());
    return mlir::reussir::success();
  }
};

class UnionAssembleOpLowering
    : public ReussirConvPatternWithLayoutCache<UnionAssembleOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      UnionAssembleOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto unionTy = cast<UnionType>(op.getResult().getType());
    const auto &layout = cache.get(unionTy);
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto one = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), getLLVMTypeConverter().getIndexType(), 1);
    auto convertedUnionTy = typeConverter->convertType(unionTy);
    auto alloca =
        rewriter.create<LLVM::AllocaOp>(op->getLoc(), ptrTy, convertedUnionTy,
                                        one, layout.getAlignment().value());
    auto tagPtr =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), ptrTy, convertedUnionTy,
                                     alloca, ArrayRef<LLVM::GEPArg>{0, 0});
    auto tagType = detail::UnionTypeImpl::getTagType(getContext(),
                                                     unionTy.getInnerTypes());
    auto dataPtr = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), ptrTy, convertedUnionTy, alloca,
        ArrayRef<LLVM::GEPArg>{0, layout.getField(1).index});
    auto tagConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), tagType, op.getTag().getZExtValue());
    rewriter.create<LLVM::StoreOp>(
        op->getLoc(), tagConst, tagPtr,
        cache.getDataLayout().getTypeABIAlignment(tagType));
    rewriter.create<LLVM::StoreOp>(op->getLoc(), adaptor.getField(), dataPtr,
                                   cache.getDataLayout().getTypeABIAlignment(
                                       adaptor.getField().getType()));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, convertedUnionTy, alloca,
                                              layout.getAlignment().value());
    return mlir::reussir::success();
  }
};

class ClosureVTableOpLowering
    : public ReussirConvPatternWithLayoutCache<ClosureVTableOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      ClosureVTableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto vtableTy =
        LLVM::LLVMStructType::getLiteral(getContext(), {ptrTy, ptrTy, ptrTy});
    auto glbOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), vtableTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        op.getName(), /*value=*/nullptr,
        /*alignment=*/cache.getDataLayout().getTypeABIAlignment(vtableTy),
        /*addrSpace=*/0, /*dsoLocal=*/true, /*threadLocal=*/false);
    Block *block = rewriter.createBlock(&glbOp.getInitializerRegion());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      mlir::Value value =
          rewriter.create<LLVM::UndefOp>(op->getLoc(), vtableTy);
      auto symbols = llvm::SmallVector<llvm::StringRef, 3>{
          adaptor.getFunc(), adaptor.getClone(), adaptor.getDrop()};
      auto enums = llvm::enumerate(symbols);
      value = std::accumulate(enums.begin(), enums.end(), value,
                              [&](mlir::Value value, auto pair) {
                                return rewriter.create<LLVM::InsertValueOp>(
                                    op->getLoc(), vtableTy, value,
                                    rewriter.create<LLVM::AddressOfOp>(
                                        op->getLoc(), ptrTy, pair.value()),
                                    pair.index());
                              });
      rewriter.create<LLVM::ReturnOp>(op->getLoc(), value);
    }
    rewriter.replaceOp(op, glbOp);
    return mlir::reussir::success();
  }
};

class FreezableVTableOpLowering
    : public ReussirConvPatternWithLayoutCache<FreezableVTableOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      FreezableVTableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto indexTy = getLLVMTypeConverter().getIndexType();
    auto vtableTy = LLVM::LLVMStructType::getLiteral(
        getContext(), {ptrTy, ptrTy, indexTy, indexTy, indexTy});
    auto glbOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), vtableTy, /*isConstant=*/true, LLVM::Linkage::LinkonceODR,
        op.getName(), /*value=*/nullptr,
        /*alignment=*/cache.getDataLayout().getTypeABIAlignment(vtableTy),
        /*addrSpace=*/0, /*dsoLocal=*/true, /*threadLocal=*/false);
    Block *block = rewriter.createBlock(&glbOp.getInitializerRegion());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      mlir::Value value =
          rewriter.create<LLVM::UndefOp>(op->getLoc(), vtableTy);
      auto symbols = std::array<FlatSymbolRefAttr, 2>{adaptor.getDropAttr(),
                                                      adaptor.getScannerAttr()};
      auto constants = std::array<mlir::Value, 3>{
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                            adaptor.getSize()),
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                            adaptor.getAlignment()),
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                            adaptor.getDataOffset())};

      auto symEnums = llvm::enumerate(symbols);
      auto constEnums = llvm::enumerate(constants);
      value = std::accumulate(
          symEnums.begin(), symEnums.end(), value,
          [&](mlir::Value value, auto pair) {
            return rewriter.create<LLVM::InsertValueOp>(
                op->getLoc(), value,
                pair.value()
                    ? rewriter
                          .create<LLVM::AddressOfOp>(op->getLoc(), ptrTy,
                                                     pair.value())
                          .getResult()
                    : rewriter.create<LLVM::ZeroOp>(op->getLoc(), ptrTy)
                          .getResult(),
                pair.index());
          });
      value = std::accumulate(constEnums.begin(), constEnums.end(), value,
                              [&](mlir::Value value, auto pair) {
                                return rewriter.create<LLVM::InsertValueOp>(
                                    op->getLoc(), value, pair.value(),
                                    pair.index() + symbols.size());
                              });
      rewriter.create<LLVM::ReturnOp>(op->getLoc(), value);
    }
    rewriter.replaceOp(op, glbOp);
    return mlir::reussir::success();
  }
};

class ClosureAssembleOpLowering
    : public ReussirConvPatternWithLayoutCache<ClosureAssembleOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      ClosureAssembleOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ty = typeConverter->convertType(op.getType());
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    mlir::Value closure = rewriter.create<LLVM::UndefOp>(op->getLoc(), ty);
    mlir::Value zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), getLLVMTypeConverter().getIndexType(), 0);
    mlir::Value vtableAddr =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), ptrTy, op.getVtable());
    closure = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), closure,
                                                   vtableAddr, 0);
    if (op.getArgpack())
      closure = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), closure,
                                                     adaptor.getArgpack(), 1);
    else {
      auto ptr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), ptrTy, zero);
      closure =
          rewriter.create<LLVM::InsertValueOp>(op.getLoc(), closure, ptr, 1);
    }
    closure =
        rewriter.create<LLVM::InsertValueOp>(op.getLoc(), closure, zero, 2);
    rewriter.replaceOp(op, closure);
    return mlir::reussir::success();
  }
};

class ProjOpLowering : public ReussirConvPatternWithLayoutCache<ProjOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      ProjOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    if (auto pointee =
            dyn_cast<CompositeType>(op.getObject().getType().getPointee())) {
      const CompositeLayout &layout = cache.get(pointee);
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, ptrTy, layout.getLLVMType(getLLVMTypeConverter()),
          adaptor.getObject(),
          llvm::ArrayRef<LLVM::GEPArg>{
              0, layout.getField(op.getIndex().getZExtValue()).index});
      return LogicalResult::success();
    }
    if (auto pointee =
            dyn_cast<ArrayType>(op.getObject().getType().getPointee())) {
      mlir::Type innerTy;
      if (pointee.getSizes().size() == 0)
        innerTy = pointee.getElementType();
      else
        innerTy = ArrayType::get(getContext(), pointee.getElementType(),
                                 pointee.getSizes().drop_front());
      auto ty = typeConverter->convertType(innerTy);
      if (!ty)
        return LogicalResult::failure();
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, ptrTy, ty, adaptor.getObject(),
          llvm::ArrayRef<LLVM::GEPArg>{op.getIndex().getZExtValue()});
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

class ValueToRefOpLowering
    : public ReussirConvPatternWithLayoutCache<ValueToRefOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      ValueToRefOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    uint64_t alignment = cache.getDataLayout().getTypePreferredAlignment(
        op.getValue().getType());
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto alloca = rewriter.create<mlir::LLVM::AllocaOp>(
        op->getLoc(), ptrTy, adaptor.getValue().getType(),
        rewriter.create<mlir::LLVM::ConstantOp>(
            op->getLoc(), getLLVMTypeConverter().getIndexType(), 1),
        alignment);
    rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), adaptor.getValue(),
                                         alloca, alignment);
    rewriter.replaceOp(op, alloca);
    return mlir::reussir::success();
  }
};

class RcBorrowOpLowering
    : public ReussirConvPatternWithLayoutCache<RcBorrowOp> {
  static inline constexpr size_t NONFREEZING_DATA_OFFSET = 1;
  static inline constexpr size_t FREEZABLE_DATA_OFFSET = 3;

public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;

  mlir::reussir::LogicalResult matchAndRewrite(
      RcBorrowOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcTy = op.getObject().getType();
    RcBoxType box =
        RcBoxType::get(getContext(), rcTy.getPointee(), rcTy.getAtomicKind(),
                       rcTy.getFreezingKind());
    const CompositeLayout &layout = cache.get(box);
    mlir::LLVM::LLVMStructType structTy =
        layout.getLLVMType(getLLVMTypeConverter());
    CompositeLayout::Field targetField = layout.getField(
        rcTy.getFreezingKind().getValue() == FreezingKind::nonfreezing
            ? NONFREEZING_DATA_OFFSET
            : FREEZABLE_DATA_OFFSET);

    // GEP to targetField.offset.
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrTy, structTy, adaptor.getObject(),
        llvm::ArrayRef<LLVM::GEPArg>{0, targetField.index});
    return mlir::reussir::success();
  }
};

class LoadOpLowering : public ReussirConvPatternWithLayoutCache<LoadOp> {
public:
  using ReussirConvPatternWithLayoutCache::ReussirConvPatternWithLayoutCache;
  mlir::reussir::LogicalResult matchAndRewrite(
      LoadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getObject(),
        cache.getDataLayout().getTypeABIAlignment(op.getType()));
    return mlir::reussir::success();
  }
};

class TokenAllocOpLowering : public mlir::OpConversionPattern<TokenAllocOp> {
public:
  using OpConversionPattern<TokenAllocOp>::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      TokenAllocOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    TokenType tokenTy = op.getToken().getType();
    const auto *cvt = static_cast<const LLVMTypeConverter *>(typeConverter);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getSize());
    auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getAlignment());
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, "__reussir_alloc", mlir::LLVM::LLVMPointerType::get(getContext()),
        mlir::ValueRange{size, alignment});
    return mlir::reussir::success();
  }
};

class NullableCheckOpLowering
    : public mlir::OpConversionPattern<NullableCheckOp> {
public:
  using OpConversionPattern<NullableCheckOp>::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      NullableCheckOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto zeroPtr = rewriter.create<mlir::LLVM::ZeroOp>(op->getLoc(), ptrTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, rewriter.getI1Type(), LLVM::ICmpPredicate::ne,
        adaptor.getNullable(), zeroPtr);
    return mlir::reussir::success();
  }
};

class RegionCleanUpOpLowering
    : public mlir::OpConversionPattern<RegionCleanUpOp> {
public:
  using OpConversionPattern<RegionCleanUpOp>::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      RegionCleanUpOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, "__reussir_clean_up_region", TypeRange{}, adaptor.getRegionCtx());
    return mlir::reussir::success();
  }
};

using NullableCoerceOpLowering = TypeCoercionLowering<NullableCoerceOp>;
using NullableNonNullOpLowering = TypeCoercionLowering<NullableNonNullOp>;
using RcTokenizeOpLowering = TypeCoercionLowering<RcTokenizeOp>;
using RcAsPtrOpLowering = TypeCoercionLowering<RcAsPtrOp>;

class NullableNullOpLowering
    : public mlir::OpConversionPattern<NullableNullOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      NullableNullOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, ptrTy);
    return mlir::reussir::success();
  }
};

class TokenFreeOpLowering : public mlir::OpConversionPattern<TokenFreeOp> {
public:
  using OpConversionPattern<TokenFreeOp>::OpConversionPattern;
  mlir::reussir::LogicalResult matchAndRewrite(
      TokenFreeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    if (auto tokenTy = dyn_cast<TokenType>(op.getToken().getType())) {
      const auto *cvt = static_cast<const LLVMTypeConverter *>(typeConverter);
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), cvt->getIndexType(), tokenTy.getSize());
      auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), cvt->getIndexType(), tokenTy.getAlignment());
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, "__reussir_dealloc", mlir::ValueRange{},
          mlir::ValueRange{adaptor.getToken(), size, alignment});
      return mlir::reussir::success();
    }
    return LogicalResult::failure();
  }
};

class UnreachableOpLowering : public mlir::OpConversionPattern<UnreachableOp> {
public:
  using OpConversionPattern<UnreachableOp>::OpConversionPattern;

  mlir::reussir::LogicalResult matchAndRewrite(
      UnreachableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    rewriter.create<func::CallOp>(op->getLoc(), "__reussir_unreachable",
                                  mlir::ValueRange{});
    if (op->getUsers().empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
          op, typeConverter->convertType(op.getType(0)));
    return mlir::reussir::success();
  }
};

class RcAcquireOpLowering : public mlir::OpConversionPattern<RcAcquireOp> {
public:
  using OpConversionPattern<RcAcquireOp>::OpConversionPattern;

  mlir::reussir::LogicalResult matchAndRewrite(
      RcAcquireOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcPtrTy = op.getRcPtr().getType();
    mlir::reussir::RcBoxType rcBoxTy =
        RcBoxType::get(getContext(), rcPtrTy.getPointee(),
                       rcPtrTy.getAtomicKind(), rcPtrTy.getFreezingKind());
    auto boxStruct = llvm::cast<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(rcBoxTy));
    auto rcTy = boxStruct.getBody()[0];
    if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::nonfreezing) {
      llvm::StringRef func =
          rcPtrTy.getAtomicKind().getValue() == AtomicKind::atomic
              ? "__reussir_acquire_atomic_freezable"
              : "__reussir_acquire_freezable";
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, func, mlir::ValueRange{}, mlir::ValueRange{adaptor.getRcPtr()});
    } else {
      auto amount =
          rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rcTy, 1);
      auto rcField = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          boxStruct, adaptor.getRcPtr(), ArrayRef<LLVM::GEPArg>{0, 0});
      if (rcPtrTy.getAtomicKind().getValue() == AtomicKind::atomic) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
            op, mlir::LLVM::AtomicBinOp::add, rcField, amount,
            mlir::LLVM::AtomicOrdering::seq_cst);
      } else {
        auto rcVal =
            rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rcTy, rcField);
        auto newRcVal =
            rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), rcTy, rcVal, amount)
                .getRes();
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, newRcVal, rcField);
      }
    }
    return mlir::reussir::success();
  }
};

class RcReleaseOpLowering : public mlir::OpConversionPattern<RcReleaseOp> {
public:
  using OpConversionPattern<RcReleaseOp>::OpConversionPattern;

  mlir::reussir::LogicalResult matchAndRewrite(
      RcReleaseOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcTy = op.getRcPtr().getType();
    // we should have already expanded nonfreezing RC pointers
    if (rcTy.getFreezingKind().getValue() == FreezingKind::nonfreezing)
      return LogicalResult::failure();
    // call the runtime function to decrease the reference count
    llvm::StringRef func = rcTy.getAtomicKind().getValue() == AtomicKind::atomic
                               ? "__reussir_release_atomic_freezable"
                               : "__reussir_release_freezable";
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, func, mlir::ValueRange{}, mlir::ValueRange{adaptor.getRcPtr()});
    return LogicalResult::success();
  }
};

class RcDecreaseOpLowering : public mlir::OpConversionPattern<RcDecreaseOp> {
public:
  using OpConversionPattern<RcDecreaseOp>::OpConversionPattern;

  mlir::reussir::LogicalResult matchAndRewrite(
      RcDecreaseOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcPtrTy = op.getRcPtr().getType();
    mlir::reussir::RcBoxType rcBoxTy =
        RcBoxType::get(getContext(), rcPtrTy.getPointee(),
                       rcPtrTy.getAtomicKind(), rcPtrTy.getFreezingKind());
    auto boxStruct = llvm::cast<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(rcBoxTy));
    auto rcTy = boxStruct.getBody()[0];
    if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::nonfreezing) {
      return LogicalResult::failure();
    } else {
      auto amount =
          rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rcTy, 1);
      auto rcField = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          boxStruct, adaptor.getRcPtr(), ArrayRef<LLVM::GEPArg>{0, 0});
      mlir::Value rcVal;
      if (rcPtrTy.getAtomicKind().getValue() == AtomicKind::atomic) {
        rcVal = rewriter.create<mlir::LLVM::AtomicRMWOp>(
            op->getLoc(), mlir::LLVM::AtomicBinOp::sub, rcField, amount,
            mlir::LLVM::AtomicOrdering::seq_cst);
      } else {
        rcVal = rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rcTy, rcField);
        auto newRcVal =
            rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), rcTy, rcVal, amount)
                .getRes();
        rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), newRcVal, rcField);
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
          op, rewriter.getI1Type(), LLVM::ICmpPredicate::eq, rcVal,
          rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), rcTy, 1));
    }
    return mlir::reussir::success();
  }
};

struct ConvertReussirToLLVMPass
    : public ConvertReussirToLLVMBase<ConvertReussirToLLVMPass> {
  using ConvertReussirToLLVMBase::ConvertReussirToLLVMBase;
  void runOnOperation() override final;
};

static void emitRuntimeFunctions(mlir::Location loc,
                                 mlir::IntegerType targetIdxTy,
                                 mlir::OpBuilder &builder) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto acquireFreezable = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_acquire_freezable"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  acquireFreezable.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  acquireFreezable->setAttr("llvm.nofree", builder.getUnitAttr());
  auto atomicAcquireFreezable = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_acquire_atomic_freezable"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  atomicAcquireFreezable.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  auto releaseFreezable = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_release_freezable"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  releaseFreezable.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  auto atomicReleaseFreezable = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_release_atomic_freezable"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  atomicReleaseFreezable.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  auto alloc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_alloc"),
      builder.getFunctionType({targetIdxTy, targetIdxTy}, {ptrTy}),
      builder.getStringAttr("private"), nullptr, nullptr);
  alloc.setArgAttr(1, "llvm.allocalign", builder.getUnitAttr());
  auto dealloc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_dealloc"),
      builder.getFunctionType({ptrTy, targetIdxTy, targetIdxTy}, {}),
      builder.getStringAttr("private"), nullptr, nullptr);
  dealloc.setArgAttr(0, "llvm.allocptr", builder.getUnitAttr());
  dealloc.setArgAttr(2, "llvm.allocalign", builder.getUnitAttr());
  auto realloc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_realloc"),
      builder.getFunctionType({ptrTy, targetIdxTy, targetIdxTy, targetIdxTy},
                              {ptrTy}),
      builder.getStringAttr("private"), nullptr, nullptr);
  realloc.setArgAttr(0, "llvm.allocptr", builder.getUnitAttr());
  realloc.setArgAttr(2, "llvm.allocalign", builder.getUnitAttr());
  auto unreachableFunc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_unreachable"),
      builder.getFunctionType({}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  unreachableFunc->setAttr(
      "llvm.linkage",
      LLVM::LinkageAttr::get(builder.getContext(),
                             LLVM::linkage::Linkage::LinkonceODR));
  unreachableFunc->setAttr("llvm.noreturn", builder.getUnitAttr());
  unreachableFunc.addEntryBlock();
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&unreachableFunc.getBlocks().front());
    builder.create<LLVM::UnreachableOp>(loc);
  }
  auto freezeFunc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_freeze"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  freezeFunc.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
  freezeFunc->setAttr("llvm.nounwind", builder.getUnitAttr());
  freezeFunc->setAttr("llvm.nofree", builder.getUnitAttr());
  freezeFunc->setAttr("llvm.mustprogress", builder.getUnitAttr());
  freezeFunc->setAttr("llvm.willreturn", builder.getUnitAttr());
  auto cleanUpRegionFunc = builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reussir_clean_up_region"),
      builder.getFunctionType({ptrTy}, {}), builder.getStringAttr("private"),
      nullptr, nullptr);
  cleanUpRegionFunc.setArgAttr(0, "llvm.nonnull", builder.getUnitAttr());
}

void ConvertReussirToLLVMPass::runOnOperation() {
  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(&module.getBodyRegion().back());
    auto targetIdxTy = builder.getIntegerType(
        dataLayout.getTypeSizeInBits(builder.getIndexType()));
    emitRuntimeFunctions(module->getLoc(), targetIdxTy, builder);
  }
  mlir::LLVMTypeConverter converter(&getContext());
  CompositeLayoutCache cache(dataLayout);
  populateLLVMTypeConverter(cache, converter);
  mlir::RewritePatternSet patterns(&getContext());
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
  patterns
      .add<RcAcquireOpLowering, RcDecreaseOpLowering, RcReleaseOpLowering,
           TokenAllocOpLowering, TokenFreeOpLowering, NullableCoerceOpLowering,
           NullableCheckOpLowering, NullableNonNullOpLowering,
           NullableNullOpLowering, RcTokenizeOpLowering, UnreachableOpLowering,
           RcAsPtrOpLowering, RegionCleanUpOpLowering>(converter,
                                                       &getContext());
  patterns
      .add<RcBorrowOpLowering, ValueToRefOpLowering, ProjOpLowering,
           LoadOpLowering, ClosureVTableOpLowering, ClosureAssembleOpLowering,
           DestroyOpLowering, UnionGetTagOpLowering, UnionInspectOpLowering,
           CompositeAssembleOpLowering, UnionAssembleOpLowering,
           RcCreateOpLowering, RegionCreateOpLowering, MRefAssignOpLowering,
           RcFreezeOpLowering, FreezableVTableOpLowering>(cache, converter,
                                                          &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalDialect<mlir::reussir::ReussirDialect>();
  llvm::SmallVector<mlir::Operation *> ops;
  module.walk([&](mlir::Operation *op) { ops.push_back(op); });
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace REUSSIR_DECL_SCOPE

namespace reussir {
std::unique_ptr<Pass> createConvertReussirToLLVMPass() {
  return std::make_unique<ConvertReussirToLLVMPass>();
}
} // namespace reussir
} // namespace mlir
