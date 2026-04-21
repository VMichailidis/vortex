#include "common.h"
#include <iostream>
#include <utility>
template <typename StructType, auto... Fields> class Kernel {
  public:
    // Kernel();
    // ~Kernel();
    template <typename... Args> static StructType apply(Args &&...args) {
        static_assert(sizeof...(Args) == sizeof...(Fields),
                      "Number of arguments must match number of fields");
        StructType _args = pack_args(args...);
        return _args;
    }

  private:
    template <typename... Args> static StructType pack_args(Args &&...args) {
        StructType result;
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ((result.*std::get<Is>(std::make_tuple(Fields...)) =
                  std::forward<Args>(args)),
             ...);
        }(std::index_sequence_for<Args...>{});
        return result;
    }
};
