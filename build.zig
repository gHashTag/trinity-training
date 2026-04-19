const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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
}
