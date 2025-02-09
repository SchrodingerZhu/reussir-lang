#ifndef REUSSIR_TYPES
#define REUSSIR_TYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "Reussir/IR/ReussirDialect.td"
include "Reussir/Interfaces/ReussirMangleInterface.td"
include "Reussir/Interfaces/ReussirCompositeLayoutInterface.td"

def AK_NonAtomic : I32EnumAttrCase<"nonatomic", 0>;
def AK_Atomic : I32EnumAttrCase<"atomic", 1>;

def FK_Nonfreezing : I32EnumAttrCase<"nonfreezing", 0>;
def FK_Frozen : I32EnumAttrCase<"frozen", 1>;
def FK_Unfrozen : I32EnumAttrCase<"unfrozen", 2>;

def AtomicKind : I32EnumAttr<
    "AtomicKind",
    "atomic kind",
    [AK_NonAtomic, AK_Atomic]> {
  let cppNamespace = "::mlir::reussir";
}

def FreezingKind : I32EnumAttr<
    "FreezingKind",
    "freezing kind",
    [FK_Nonfreezing, FK_Frozen, FK_Unfrozen]> {
  let cppNamespace = "::mlir::reussir";
}

class Reussir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Reussir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Reussir_IntType : AnyTypeOf<[
    Builtin_Index, Builtin_Integer
]>;

def Reussir_FPType : AnyTypeOf<[
    Builtin_Float16, Builtin_Float32, Builtin_Float64, Builtin_Float128
]>;

def Reussir_TokenType : Reussir_Type<"Token", "token",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReussirMangleInterface>]> {
  let summary = "Reussir Memory Token";
  let description = [{
    `reussir.token` is the token of the memory
  }];
  let parameters = (ins
    "size_t":$alignment,
    "size_t":$size
  );
  let assemblyFormat = [{
    `<` `size` `:` $size `,` `alignment` `:` $alignment `>`
  }];
  let genVerifyDecl = 1;
}

def Reussir_RegionCtxType : Reussir_Type<"RegionCtx", "region_ctx",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Region Context";
  let description = [{
    `reussir.region_ctx` is used to track mutable subregions.
    It contains the root for regional garbage collection.
    Region {
      FreezableRcHeader* tail;
    }
    RegionCtx is a pointer to Region. For now, it can be understood as a pointer to a pointer.
  }];
}

def Reussir_NullableType : Reussir_Type<"Nullable", "nullable",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir Nullable Type";
  let description = [{
    `reussir.nullable` is a nullable type. Rc pointers and tokens on default at nonnull in Reussir.
    This is just a type wrapper to indicate that the pointer can be null.
  }];
  let parameters = (ins
    "::mlir::Type":$pointer
  );
  let assemblyFormat = [{
    `<` $pointer `>`
  }];
}

def Reussir_RcBoxType : Reussir_Type<"RcBox", "rcbox",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReussirCompositeLayoutInterface>
    ]> {
  let summary = "Reussir Control Box";
  let description = [{
    `reussir.rcbox` is the underlying control box of RC managed objects.
    It has the following varaints:
    - Non-atomic plain RC
    - Non-atomic freezable RC
    - Atomic plain RC
    - Atomic freezable RC
    For Non-freezable variants, status is simply an index type:
    ```
    RcBox<T> {
        size_t status;
        T data;
    };
    ```
    For freezable varaints, status is also of the same size but it is actually 
    a tagged pointer. Please refer to https://dl.acm.org/doi/abs/10.1145/3652024.3665507
    for more details:
    ```
    RcBox<T> {
        union {
          size_t status;
          RcBox* root;
        } status;
        RcBox* next;
        void* vtable;
        T data;
    };
    struct VTable {
      void*  drop;
      size_t size;
      size_t alignment;
      ssize_t scan_count;
      size_t scan_offset[0];
    }
    - if scan_count >= 0, then scan_offset stores offsets for inner mutable fields (composite types).;
    - if scan_count < 0, then scan_offset contains a single value annotates the stride between inner mutable fields (array types).
    ```
  }];
  let parameters = (ins
    "mlir::Type":$dataType,
    "::mlir::reussir::AtomicKindAttr":$atomicKind,
    "::mlir::reussir::FreezingKindAttr":$freezingKind
  );
  let assemblyFormat = [{
    `<` $dataType `,` custom<AtomicKind>($atomicKind) `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def Reussir_RcType : Reussir_Type<"Rc", "rc",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReussirMangleInterface>]> {

  let summary = "Reussir Rc Pointer Type";
  let description = [{
    `reussir.rc` is the Rc pointer to managed object (in box)
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reussir::AtomicKindAttr":$atomicKind,
    "::mlir::reussir::FreezingKindAttr":$freezingKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<AtomicKind>($atomicKind) `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def Reussir_MRefType : Reussir_Type<"MRef", "mref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "Reussir Mutable Reference Type";
  let description = [{
    `reussir.mref` is the mutable reference that is assignable inside a unfrozen but freezable object.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reussir::AtomicKindAttr":$atomicKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<AtomicKind>($atomicKind) `>`
  }];
}

def Reussir_RefType : Reussir_Type<"Ref", "ref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "Reussir Reference Type";
  let description = [{
    `reussir.ref` is the referece to the underlying type. It serves as the pointer
    in Reussir.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reussir::FreezingKindAttr":$freezingKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def Reussir_ArrayType : Reussir_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReussirMangleInterface>]> {

  let summary = "Reussir Array Type";
  let description = [{
    `reussir.array` is a multi-dimensional array with fixed lengths
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"size_t">:$sizes
  );

  let assemblyFormat = [{
    `<` $elementType `,` $sizes `>`
  }];
}

def Reussir_VectorType : Reussir_Type<"Vector", "vector",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReussirCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReussirMangleInterface>
    ]> {
  let summary = "Reussir Vector Type";
  let description = [{
    `reussir.vector` is a single-dimension vector with dynamic length.
    Vec<T> {
      T* data;
      size_t capacity;
      size_t length;
    }
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

def Reussir_OpaqueType : Reussir_Type<"Opaque", "opaque",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReussirCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReussirMangleInterface>
    ]> {
  let summary = "Reussir Opaque Type";
  let description = [{
    `reussir.opaque` is an opaque type for externally managed object.
    Opaque {
      void* drop;
      void* clone;
      T data;
    }
  }];

  let parameters = (ins
    "mlir::IntegerAttr":$size,
    "mlir::IntegerAttr":$alignment
  );

  let assemblyFormat = [{
    `<` $size `,` $alignment `>`
  }];
}


def Reussir_CompositeType : Type<CPred<"::mlir::isa<::mlir::reussir::CompositeType>($_self)">,
                          "Reussir composite type">;

def Reussir_UnionType : Type<CPred<"::mlir::isa<::mlir::reussir::UnionType>($_self)">,
                          "Reussir union  type">;

def Reussir_ClosureType : Reussir_Type<"Closure", "closure",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, 
      DeclareTypeInterfaceMethods<ReussirCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReussirMangleInterface>
    ]> {

  let summary = "Reussir Closure Type";
  let description = [{
    `reussir.closure` is for capturing lambda expressions.
    struct Closure {
      void* vtable;
      void* arg_start;
      void* arg_cursor;
    };
    struct VTable {
      void* func;
      void* clone;
      void* drop;
    }
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputTypes,
    OptionalParameter<"::mlir::Type">:$outputType
  );

  let assemblyFormat = [{
    `<` `(` $inputTypes `)` (`->` $outputType^)? `>`
  }];
}

class Reussir_RcTypeOf<list<Type> allowedTypes> : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointee) { return ",
         SubstLeaves<"$_self", "pointee", AnyTypeOf<allowedTypes>.predicate>,
         "; }(::llvm::cast<::mlir::reussir::RcType>($_self).getPointee())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reussir::RcType"
>;

class Reussir_RefTypeOf<list<Type> allowedTypes> : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointee) { return ",
         SubstLeaves<"$_self", "pointee", AnyTypeOf<allowedTypes>.predicate>,
         "; }(::llvm::cast<::mlir::reussir::RefType>($_self).getPointee())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reussir::RefType"
>;

def Reussir_NullableToken : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointer) { return ",
         SubstLeaves<"$_self", "pointer", AnyTypeOf<[Reussir_TokenType]>.predicate>,
         "; }(::llvm::cast<::mlir::reussir::NullableType>($_self).getPointer())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reussir::NullableType"
>;

def Reussir_PtrType : Reussir_Type<"Ptr", "ptr",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Reussir opaque pointer type";
  let description = [{
    `reussir.ptr` is an opaque pointer type.
  }];
}

#endif // REUSSIR_TYPES
