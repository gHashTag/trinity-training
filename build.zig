const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build modules
    const railway_api = b.createModule(.{
        .root_source_file = b.path("src/tri/railway_api.zig"),
        .target = target,
        .optimize = optimize,
    });

    const railway_circuit = b.createModule(.{
        .root_source_file = b.path("src/tri/railway_circuit_breaker.zig"),
        .target = target,
        .optimize = optimize,
    });

    const railway_farm = b.createModule(.{
        .root_source_file = b.path("src/tri/railway_farm.zig"),
        .target = target,
        .optimize = optimize,
    });
    railway_farm.addImport("railway_api", railway_api);
    railway_farm.addImport("railway_circuit", railway_circuit);

    const cloud_train = b.createModule(.{
        .root_source_file = b.path("src/tri/cloud_train.zig"),
        .target = target,
        .optimize = optimize,
    });
    cloud_train.addImport("railway_api", railway_api);
    cloud_train.addImport("railway_farm", railway_farm);

    const hslm = b.createModule(.{
        .root_source_file = b.path("src/hslm/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    hslm.addImport("railway_api", railway_api);
    hslm.addImport("railway_farm", railway_farm);

    const hslm_exe = b.addExecutable(.{
        .name = "hslm-cli",
        .root_module = hslm,
    });

    const train_step = b.step("train", "Build HSLM CLI");
    train_step.dependOn(&hslm_exe.step);
    b.default_step = train_step;

    // There was no test step at all. `zig build test` answered
    // "error: no step named 'test'", while 672 `test` blocks sat in the tree
    // across 104 files. Nothing could run them, so nothing ever had.
    const test_step = b.step("test", "Run tests");
    const test_roots = [_]struct { name: []const u8, mod: *std.Build.Module }{
        .{ .name = "hslm", .mod = hslm },
        .{ .name = "railway_farm", .mod = railway_farm },
        .{ .name = "cloud_train", .mod = cloud_train },
        .{ .name = "railway_api", .mod = railway_api },
        .{ .name = "railway_circuit", .mod = railway_circuit },
    };
    for (test_roots) |r| {
        const t = b.addTest(.{ .name = r.name, .root_module = r.mod });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
