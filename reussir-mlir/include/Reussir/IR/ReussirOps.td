#ifndef REUSSIR_OPS
#define REUSSIR_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinAttributes.td"

include "Reussir/IR/ReussirDialect.td"
include "Reussir/IR/ReussirAttrs.td"
include "Reussir/IR/ReussirTypes.td"

class Reussir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Reussir_Dialect, mnemonic, traits>;

def TokenAllocOp : Reussir_Op<"token.alloc", []> {

  let summary = "Allocate memory token";
  let description = [{
    `reussir.alloc` allocates a memory token.
    ```mlir
    reussir.alloc : !reussir.token<size: 16, alignment: 8>
    ```
  }];

  let results = (outs Res<
    Reussir_TokenType,
    "Allocated Memory Token", [MemAlloc<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `:` type($token) attr-dict
  }];
}

def TokenFreeOp : Reussir_Op<"token.free", []> {
  let summary = "Free memory token";
  let description = [{
    `reussir.token.free` deallocates a memory token.
    ```mlir
    reussir.token.free (%x : !reussir.token<size: 16, alignment: 8>)
    ```
  }];

  let arguments = (ins Res<
    AnyTypeOf<[Reussir_TokenType, Reussir_NullableToken]>,
    "Free Memory Token", [MemFree<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict
  }];
}

def TokenReinterpretOp : Reussir_Op<"token.reinterpret"> {
  let summary = "Reinterpret the token type";
  let description = [{
    `reussir.token.reinterpret` reinterprets the token type.
  }];
  let results = (outs Res<Reussir_RefType, "reinterpreted reference">:$result);
  let arguments = (ins Arg<Reussir_TokenType, "Token to reinterpret">:$token);
  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict `:` type($result)
  }];
}

def TokenEnsureOp : Reussir_Op<"token.ensure"> {
  let summary = "Ensure a nonnull token";
  let description = [{
    `reussir.token.ensure` converts a possibly nullable token to a token by checking
    the nullability and allocate memory on need.
  }];
  let results = (outs Res<Reussir_TokenType, "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments = (ins Arg<Reussir_NullableToken, "input token", [MemFree<DefaultResource>]>:$nullableToken);
  let assemblyFormat = [{
    `(` $nullableToken `:` type($nullableToken) `)` attr-dict `:` type($result)
  }];
}

def TokenReallocOp : Reussir_Op<"token.realloc"> {
  let summary = "Ensure a nonnull token";
  let description = [{
    `reussir.token.realloc` converts a possibly nullable token to a token by checking
    the nullability and reallocate the memory on need. Utimately, we should custom the
    allocator to avoid memcpy if the content is not to be retained.
  }];
  let results = (outs Res<Reussir_TokenType, "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments = (ins Arg<Reussir_TokenType, "input token", [MemFree<DefaultResource>]>:$token);
  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict `->` type($result)
  }];
}

def NullableCheckOp : Reussir_Op<"nullable.check"> {
  let summary = "Check the nullibility of a pointer";
  let description = [{
    `reussir.nullable.check` returns 1 if target pointer is null or 0 otherwise.
  }];
  let results = (outs Res<I1, "result">:$result);
  let arguments = (ins Arg<Reussir_NullableType, "input">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` attr-dict `->` type($result)
  }];
}

def NullableCoerceOp : Reussir_Op<"nullable.coerce"> {
  let summary = "Force unwrap the nullable as a nonnull pointer";
  let description = [{
    `reussir.nullable.coerce` forcibly unwrap the nullable as a nonnull pointer.
  }];
  let results = (outs Res<AnyTypeOf<[Reussir_RcType, Reussir_TokenType]>, "result">:$result);
  let arguments = (ins Arg<Reussir_NullableType, "input token">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` attr-dict `:` type($result)
  }];
}

def NullableNullOp : Reussir_Op<"nullable.null"> {
  let summary = "Create a null pointer";
  let description = [{
    `reussir.nullable.null` creates a null pointer.
  }];
  let results = (outs Res<Reussir_NullableType, "result">:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def NullableNonNullOp : Reussir_Op<"nullable.nonnull"> {
  let summary = "Create a nullable pointer from a nonnull value";
  let description = [{
    `reussir.nullable.nonnull` creates a nullable pointer from a nonnull value.
  }];
  let results = (outs Res<Reussir_NullableType, "result">:$result);
  let arguments = (ins Arg<AnyTypeOf<[Reussir_RcType, Reussir_TokenType]>, "pointer">:$pointer);
  let assemblyFormat = [{
    `(` $pointer `:` type($pointer) `)` attr-dict `:` type($result)
  }];
}

def RcAcquireOp : Reussir_Op<"rc.acquire"> {

  let summary = "Increase reference count";
  let description = [{
    `reussir.rc.acquire` increases the reference count of the object.
    ```mlir
    reussir.rc.acquire (%0 : !reussir.rc<i32>)
    ```
  }];

  let arguments = (ins Arg<Reussir_RcType, "target rc pointer", [MemReadAt<0>, MemWriteAt<1>]>:$rcPtr);

  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` attr-dict
  }];

  let hasVerifier = 1;
}

def RcDecreaseOp : Reussir_Op<"rc.decrease"> {
  let summary = "Decrease reference count";
  let description = [{
    This decrement operation is for progressively lowering rc.release operation. It decreases
    the reference count and return a flag indicating whether the object should be dropped.
  }];

  let arguments = (ins Arg<Reussir_RcType, "target rc pointer", [MemReadAt<0>, MemWriteAt<1>]>:$rcPtr);
  let results = (outs Res<I1, "drop flag">:$dropFlag);
  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` attr-dict `:`  type($dropFlag) 
  }];

  let hasVerifier = 1;
}

def RcReleaseOp : Reussir_Op<"rc.release"> {
  let summary = "Release reference count and drop the object if necessary";
  let description = [{
    `reussir.rc.release` decreases the reference count of the object.
    - It target rc is a freezable one, it has no result. The operation translates to runtime function call.
    - If the target rc is not freezable, it returns a nullable token.
  }];

  let arguments = (ins 
    Arg<Reussir_RcType, "target rc pointer", [MemReadAt<0>, MemWriteAt<1>]>:$rcPtr,
    OptionalAttr<IndexAttr>:$tag
  );
  let results = (outs Res<Optional<Reussir_NullableToken>, "token">:$token);
  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` (`tag` `(` $tag^ `)`)? attr-dict  (`:` type($token)^)?
  }];

  let hasVerifier = 1;
}

def ProjectableType : AnyTypeOf<[Reussir_RcType, Reussir_RefType, Reussir_CompositeType, Reussir_ArrayType]>;

def ValueToRefOp : Reussir_Op<"val2ref"> {
    let summary = "Promote a value to stack and get the reference";
    let description = [{
      `reussir.val2ref` is used to promote a value to an alloca on stack.
      For composite/union/array arguments passed as values:
        - caller should either use `byval` ptr call convention: that is, spill the
          value to stack pass its ref.
        - or when the struct is small enough, the value can be directly passed to
          target function but inorder to decompose the the object, this operaton is
          still needed.
    }];
    let results = (outs Res<Reussir_RefType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);
    let arguments = (
        ins AnyType:$value
    );
    let assemblyFormat = [{
      $value attr-dict `:` type($value) `->` type($result)
    }];
    let hasVerifier = 1;
}

def RcBorrowOp : Reussir_Op<"rc.borrow"> {
    let summary = "Borrow the managed object and get the reference";
    let description = [{
      `reussir.rc.borrow` is used to get the reference to the managed object.
    }];
    let results = (outs Reussir_RefType:$result);
    let arguments = (
        ins Reussir_RcType:$object
    );
    let assemblyFormat = [{
      $object attr-dict `:` type($object) `->` type($result)
    }];
    let hasVerifier = 1;
}

def RcTokenizeOp : Reussir_Op<"rc.tokenize"> {
    let summary = "Convert the rc to a token that is suitable for its underlying RcBox";
    let description = [{
      `reussir.rc.tokenize` converts the rc to a token that is suitable for its underlying RcBox.
    }];
    let results = (outs Reussir_TokenType:$token);
    let arguments = (
        ins Reussir_RcType:$object
    );
    let assemblyFormat = [{
      $object attr-dict `:` type($object) `->` type($token)
    }];
}

def ProjOp : Reussir_Op<"proj"> {
  let summary = "Project a field out of a value type or a referential type";
  let description = [{
    `reussir.proj` gets a reference to the field of the object similar to the
    GEP operation in LLVM IR without the first index.
    The freezing attribute is inherited from the parent reference.
  }];
  let results = (outs Reussir_RefType:$result);
  let arguments = (
    ins Arg<Reussir_RefType, "reference to the object", []>:$object,
    IndexAttr:$index
  );

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict `:` type($object) `->` type($result)
  }];
  let hasVerifier = 1;
}

def LoadOp : Reussir_Op<"load"> {
  let summary = "Load a value from a reference";
  let description = [{
    `reussir.load` loads the value from the reference.
    One special thing is the mref type, which will be
    loaded as a `reussir.rc` type, whose freezing state
    inherits from the parent reference.
  }];
  let results = (outs AnyType:$result);
  let arguments = (
    ins Arg<Reussir_RefType, "reference to the object", [MemRead]>:$object
  );

  let assemblyFormat = [{
    $object attr-dict `:` type($object) `->` type($result)
  }];
  let hasVerifier = 1;
}

def DestroyOp : Reussir_Op<"destroy"> {
  let summary = "Destroy a value type or a referential type";
  let description = [{
    - For composite/union type (either a reference or the value), 
      this operation project out the fields that require additional 
      maintenance and apply the required operations.
      - For RC type, it applies the `reussir.rc.release` operation.
      - For nested composite/union types, it recursively applies the 
        `reussir.destroy` operation.
    - Specialized types such as Opaque/Vector/Closure are specially handled.
  }];

  // TODO: Add type restrictions
  let arguments = (ins 
    Arg<Reussir_RefType, "destroyable object", [MemRead, MemFree<DefaultResource>]>:$object,
    OptionalAttr<IndexAttr>:$tag
  );

  let assemblyFormat = [{
    `(` $object `:` type($object) `)` (`tag` `(` $tag^ `)`)? attr-dict
  }];
}

def CloneOp : Reussir_Op<"clone"> {
  let summary = "Make sure the object behind the reference can be value-copied";
  let description = [{
    `reussir.clone` applies shadow copy to the object behind the reference. That is,
    it will increase the reference count of sub fields if necessary.
  }];

  let arguments = (ins 
    Arg<Reussir_RefType, "clonable object">:$object,
    OptionalAttr<IndexAttr>:$tag
  );

  let results = (outs Res<AnyType, "Cloned Object">:$result);

  let assemblyFormat = [{
    `(` $object `:` type($object) `)` (`tag` `(` $tag^ `)`)? attr-dict `:` type($result)
  }];
}

def UnionGetTagOp : Reussir_Op<"union.get_tag"> {
  let summary = "Get the tag of the union type";
  let description = [{
    `reussir.union.get_tag` gets the tag of the union type.
  }];
  let results = (outs Builtin_Index:$tag);
  let arguments = (ins Reussir_RefTypeOf<[Reussir_UnionType]>:$unionRef);
  let assemblyFormat = [{
    $unionRef attr-dict `:` type($unionRef) `->` type($tag)
  }];
}

def UnionInspectOp : Reussir_Op<"union.inspect"> {
  let summary = "Inspect the union type";
  let description = [{
    `reussir.union.inspect` inspects the union type and returns a reference to the type of
    the given index. Such inspection is not checked in runtime.
  }];
  let results = (outs Optional<Reussir_RefType>:$result);
  let arguments = (ins Reussir_RefTypeOf<[Reussir_UnionType]>:$unionRef, IndexAttr:$index);
  let assemblyFormat = [{
    $unionRef `[` $index `]` attr-dict `:` type($unionRef) (`->` type($result)^)?
  }];
}

def PanicOp : Reussir_Op<"panic"> {
  let summary = "Panic the program";
  let description = [{
    `reussir.panic` panics the program.
  }];
  let arguments = (ins Builtin_StringAttr:$message);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    `(` $message `)` attr-dict (`:` type($result)^)?
  }];
}

def UnreachableOp : Reussir_Op<"unreachable"> {
  let summary = "Notify the program that the code path is unreachable";
  let description = [{
    `reussir.unreachable` notifies the program that the code path is unreachable.
  }];
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    attr-dict (`:` type($result)^)?
  }];
}

def ClosureNewOp : Reussir_Op<"closure.new"> {
  let summary = "Create a new closure";
  let description = [{
    `reussir.closure.new` creates a new closure object (without wrapping it into RC). 
    The closure shall not capture mutable references (for now).
  }];
  let results = (outs Res<Reussir_ClosureType, 
    "Closure Object", [MemAlloc<DefaultResource>]>:$closure);
  let regions = (region VariadicRegion<MinSizedRegion<1>>:$body);
  let assemblyFormat = [{
    ($body^)? `:` type($closure) attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::reussir::ClosureType getClosureType();
  }];
  // TODO: add verifier
  let hasVerifier = 1;
}

def ClosureAssembleOp : Reussir_Op<"closure.assemble"> {
  let summary = "Create a new closure with given resources";
  let description = [{
    `reussir.closure.assemble` creates a new closure object. 
    The closure can either be a value or wrapped in a rc pointer.
    The closure shall not capture mutable references (for now).
  }];
  let results = (outs Res<Reussir_ClosureType, 
    "Closure Object", [MemAlloc<DefaultResource>]>:$closure);
  let arguments = (ins FlatSymbolRefAttr:$vtable, 
    Arg<Optional<Reussir_TokenType>, "memory resource for argpack", [MemWrite]>:$argpack);
  let assemblyFormat = [{
    `vtable` `(` $vtable `)`
    (`argpack` `(` $argpack^ `:` type($argpack) `)` )?
    `:` attr-dict type($closure)
  }];
}

def ClosureYieldOp : Reussir_Op<"closure.yield", [ReturnLike, Terminator,
    ParentOneOf<["::mlir::reussir::ClosureNewOp"]>]> {
  let summary = "Yield closure return value";
  let description = [{
    `closure.yield` yields the return value of the closure.
    This serves as the terminator of the closure body.
  }];

  let arguments = (ins Optional<AnyType>:$value);

  let assemblyFormat = [{
   ($value^ `:` type($value))? attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 1;
}

def ClosureVTableOp : Reussir_Op<"closure.vtable"> {
  let summary = "Define a closure vtable";
  let description = [{
    `closure.vtable` defines a vtable for the closure.
    The vtable is a list of function pointers.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name, TypeAttr:$closureType, FlatSymbolRefAttr:$func, FlatSymbolRefAttr:$clone, FlatSymbolRefAttr:$drop);

  let assemblyFormat = [{
    attr-dict $name `{` 
      `closure_type` `:` $closureType `,`
      `func` `:` $func `,`
      `clone` `:` $clone `,`
      `drop` `:` $drop
    `}` 
  }];
}

def ClosureApplyOp : Reussir_Op<"closure.apply"> {
  let summary = "Apply one more argument to the closure";
  let description = [{
    `closure.apply` adds one more argument to the closure's argpack.
    This operation "consumes" both the closure and the argument. The RC shall be increased
    if relevant objects are still to be used after the operation.
  }];
  let results = (outs Res<Reussir_RcTypeOf<[Reussir_ClosureType]>, "Result">:$result);
  let arguments = (ins Reussir_RcTypeOf<[Reussir_ClosureType]>:$closure, Variadic<AnyType>:$args);
  let assemblyFormat = [{
    $closure `(` $args `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def ClosureEvalOp : Reussir_Op<"closure.eval"> {
  let summary = "Evaluate a fully applied closure";
  let description = [{
    `closure.eval` evaluates a fully applied closure. 
    The operation consumes the closure.
  }];
  let results = (outs Res<AnyType, "Result">:$result);
  let arguments = (ins Reussir_RcTypeOf<[Reussir_ClosureType]>:$closure);
  let assemblyFormat = [{
    `(` $closure `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def RcUniquifyOp : Reussir_Op<"rc.uniquify"> {
  let summary = "Make sure the underlying object is unique";
  let description = [{
    `rc.uniquify` makes sure the underlying object is unique.
    - If the count is 1, it returns the object itself.
    - Otherwise, it clones the object and returns the new object. 
      The original rc is dropped.
  }];
  let results = (outs Res<Reussir_RcType, "Unique Rc Object">:$result);
  let arguments = (ins Reussir_RcType:$rc);
  let assemblyFormat = [{
    `(` $rc `:` type($rc) `)` attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def RcIsUniqueOp : Reussir_Op<"rc.is_unique"> {
  let summary = "Check if the underlying object is unique";
  let description = [{
    `rc.is_unique` checks if the underlying object is unique.
  }];
  let results = (outs Res<I1, "Unique Flag">:$result);
  let arguments = (ins Reussir_RcType:$rc);
  let assemblyFormat = [{
    `(` $rc `:` type($rc) `)` attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def RcAsPtrOp : Reussir_Op<"rc.as_ptr"> {
  let summary = "Cast rc pointer to LLVM opaque pointer";
  let description = [{
    `rc.as_ptr` casts the rc pointer to LLVM opaque pointer.
  }];
  let results = (outs Res<Reussir_PtrType, "Opaque Pointer">:$result);
  let arguments = (ins Reussir_RcType:$rc);
  let assemblyFormat = [{
    `(` $rc `:` type($rc) `)` attr-dict `:` type($result)
  }];
}

def FreezableVTableOp : Reussir_Op<"rc.fvtable"> {
  let summary = "Define a vtable for freezable rc objects";
  let description = [{
    `rc.fvtable` defines a vtable for the freezable rc objects.
  }];

  let arguments = (ins 
    FlatSymbolRefAttr:$name, 
    OptionalAttr<FlatSymbolRefAttr>:$drop, 
    OptionalAttr<FlatSymbolRefAttr>:$scanner,
    IndexAttr:$size, 
    IndexAttr:$alignment,
    IndexAttr:$dataOffset
  );

  let assemblyFormat = [{
    attr-dict $name `{` 
      (`drop` `:` $drop^ `,`)?
      (`scanner` `:` $scanner^ `,`)?
      `size` `:` $size `,`
      `alignment` `:` $alignment `,`
      `data_offset` `:` $dataOffset
    `}` 
  }];
}

def CompositeAssembleOp : Reussir_Op<"composite.assemble"> {
  let summary = "Create a new composite object";
  let description = [{
    `reussir.composite.assemble` creates a new composite object.
  }];
  let results = (outs Res<Reussir_CompositeType, "Composite Object">:$composite);
  let arguments = (ins Variadic<AnyType>:$fields);
  let assemblyFormat = [{
    `(` $fields `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def UnionAssembleOp : Reussir_Op<"union.assemble"> {
  let summary = "Create a new union object";
  let description = [{
    `reussir.union.assemble` creates a new union object.
  }];
  let results = (outs Res<Reussir_UnionType, "Union Object">:$result);
  let arguments = (ins Arg<IndexAttr, "tag">:$tag, AnyType:$field);
  let assemblyFormat = [{
    `(` $tag `,` $field `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def RcCreateOp : Reussir_Op<"rc.create"> {
  let summary = "Create a new Rc object";
  let description = [{
    `reussir.rc.create` creates a new Rc object.
  }];
  let results = (outs Res<Reussir_RcType, "Rc Object", [MemAlloc<DefaultResource>]>:$rc);
  let arguments = (ins 
    AnyType:$value, 
    Reussir_TokenType:$token,
    Optional<Reussir_RegionCtxType>:$region
  );
  let assemblyFormat = [{
    `value` `(` $value `)`
    `token` `(` $token `)`
    (`region` `(` $region^ `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def RegionRunOp : Reussir_Op<"region.run", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "Create a new region";
  let description = [{
    `reussir.region.run` creates a new region.
  }];
  let results = (outs Res<Optional<AnyType>, "Result">:$result);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    ($body^)? attr-dict (`:` type($result)^)?
  }];
  let hasVerifier = 1;
}

def RegionCreateOp : Reussir_Op<"region.create"> {
  let summary = "Create a new region ctx";
  let description = [{
    `reussir.region.create` creates a new region ctx. It serves as an intermediate placeholder for lowering `region.run` operation.
  }];
  let results = (outs Res<Reussir_RegionCtxType, "res">:$regionCtx);
  let assemblyFormat = [{
    attr-dict `:` type($regionCtx)
  }];
}

def RegionCleanUpOp : Reussir_Op<"region.cleanup"> {
  let summary = "Clean up the region ctx";
  let description = [{
    `reussir.region.cleanup` cleans up the region ctx. It serves as an intermediate placeholder for lowering `region.run` operation.
  }];
  let arguments = (ins Reussir_RegionCtxType:$regionCtx);
  let assemblyFormat = [{
    `(` $regionCtx `:` type($regionCtx) `)` attr-dict
  }];
}

def RegionYieldOp : Reussir_Op<"region.yield", [
    ReturnLike, 
    Terminator,
    ParentOneOf<["::mlir::reussir::RegionRunOp"]>]> {
  let summary = "Yield region return value";
  let description = [{
    `region.yield` yields the return value of the region.
    This serves as the terminator of the region body. When a region exists, all objects 
    are scanned from the tail pointer. The freeze operation serves as a marking process. 
    All other regional objects that are not marked are recycled.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = [{
   $value attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

def MRefAssignOp : Reussir_Op<"mref.assign"> {
  let summary = "Assign a value to a mutable reference";
  let description = [{
    `reussir.mref.assign` assigns a value to a mutable reference.
  }];
  let arguments = (ins 
    Arg<Reussir_RefTypeOf<[Reussir_MRefType]>, "target mref pointer", [MemWrite]>:$refOfMRef, 
    Reussir_NullableType:$value
  );
  let assemblyFormat = [{
    $value `to` $refOfMRef attr-dict `:` type($value) `,` type($refOfMRef)
  }];
  let hasVerifier = 1;
}

def RcFreezeOp : Reussir_Op<"rc.freeze"> {
  let summary = "Freeze the Rc object";
  let description = [{
    `reussir.rc.freeze` freezes the Rc object. Limited by the control flow interface of the MLIR, the RegionYieldOp must have consistent type
    with its parent RegionRunOp. Hence, we cannot automatically freeze the Rc object. This operation serves a marker for such conversion. It is up to
    the user to make sure that this operation is applied only for the use of yielding the Rc object. Object after being frozen shall not be modified.
  }];
  let arguments = (ins Reussir_RcType:$rcPtr);
  let results = (outs Res<Reussir_RcType, "Frozen Rc Object">:$result);
  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

#endif // REUSSIR_OPS
