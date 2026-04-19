const std = @import("std");
pub fn build(b: *std.Build) !void {
    _ = b.addExecutable(.{
        .name = "hslm-cli",
        .root_source_file = .{ .path = "src/hslm/cli.zig" },
    });
}