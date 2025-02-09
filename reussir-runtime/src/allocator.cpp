#include "snmalloc/snmalloc.h"

extern "C" [[gnu::visibility("default")]] void *
__reussir_alloc_impl(size_t size, size_t alignment) {
  using namespace snmalloc;
  return ThreadAlloc::get().alloc(aligned_size(alignment, size));
}

extern "C" [[gnu::visibility("default")]] void
__reussir_dealloc_impl(void *ptr, size_t size, size_t alignment) {
  using namespace snmalloc;
  ThreadAlloc::get().dealloc(ptr, aligned_size(alignment, size));
}

extern "C" [[gnu::visibility("default")]] void *
__reussir_realloc_impl(void *ptr, size_t old_size, size_t alignment,
                       size_t new_size) {
  using namespace snmalloc;
  size_t aligned_old_size = aligned_size(alignment, old_size),
         aligned_new_size = aligned_size(alignment, new_size);
  if (size_to_sizeclass_full(aligned_old_size).raw() ==
      size_to_sizeclass_full(aligned_new_size).raw())
    return ptr;
  void *p = ThreadAlloc::get().alloc(aligned_new_size);
  if (p) {
    std::memcpy(p, ptr, old_size < new_size ? old_size : new_size);
    ThreadAlloc::get().dealloc(ptr, aligned_old_size);
  }
  return p;
}

extern "C" [[gnu::visibility("default")]] void *
__reussir_realloc_nocopy_impl(void *ptr, size_t old_size, size_t alignment,
                              size_t new_size) {
  using namespace snmalloc;
  size_t aligned_old_size = aligned_size(alignment, old_size),
         aligned_new_size = aligned_size(alignment, new_size);
  if (size_to_sizeclass_full(aligned_old_size).raw() ==
      size_to_sizeclass_full(aligned_new_size).raw())
    return ptr;
  ThreadAlloc::get().dealloc(ptr, aligned_old_size);
  return ThreadAlloc::get().alloc(aligned_new_size);
}
