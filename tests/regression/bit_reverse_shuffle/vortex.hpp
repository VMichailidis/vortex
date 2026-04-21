namespace vx {
template <typename StructType, auto... Fields> class Kernel {
  public:
    Kernel();
    ~Kernel();
    template <typename... Args> static void apply(Args &&...args) {
        StructType _args = pack_args(args...);
    }

  private:
    template <typename... Args> static StructType pack_args(Args &&...args) {
        printf(args...);
    }
};
} // namespace vx
