pub const packages = struct {
    pub const @"golden_float-0.2.0-h7LKhdEXCgA666AjkuikpycwldAc3ftwT5F-pn3NtfLn" = struct {
        pub const build_root = "/Users/playra/trinity-training/zig-pkg/golden_float-0.2.0-h7LKhdEXCgA666AjkuikpycwldAc3ftwT5F-pn3NtfLn";
        pub const build_zig = @import("golden_float-0.2.0-h7LKhdEXCgA666AjkuikpycwldAc3ftwT5F-pn3NtfLn");
        pub const deps: []const struct { []const u8, []const u8 } = &.{
        };
    };
};

pub const root_deps: []const struct { []const u8, []const u8 } = &.{
    .{ "zig_golden_float", "golden_float-0.2.0-h7LKhdEXCgA666AjkuikpycwldAc3ftwT5F-pn3NtfLn" },
};
