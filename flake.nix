{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    crane.url = "github:ipetkov/crane";
    crane.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, fenix, crane, ... }:
    let
      # Supported systems
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      # Helper to generate outputs for each system
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          rustToolchain = fenix.packages.${system}.complete;
        in
        {
          default = pkgs.mkShell {
            packages = [
              rustToolchain.toolchain
              pkgs.pkg-config
              pkgs.openssl
              pkgs.wayland
              pkgs.wayland-protocols
              pkgs.libxkbcommon
              pkgs.libdecor
              pkgs.vulkan-loader
              pkgs.llvmPackages.libclang
              pkgs.linuxHeaders
            ];

            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include:${pkgs.glibc.dev}/include";

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.wayland
              pkgs.libdecor
              pkgs.libxkbcommon
              pkgs.vulkan-loader
              pkgs.llvmPackages.libclang.lib
            ];
          };
        }
      );

      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          rustToolchain = fenix.packages.${system}.complete.toolchain;
          craneLib = crane.lib.${system}.overrideToolchain rustToolchain;

          # Common arguments for crane
          commonArgs = {
            src = craneLib.cleanCargoSource (craneLib.path ./.);
            strictDeps = true;
            nativeBuildInputs = with pkgs; [
              pkg-config
              openssl
              wayland
              wayland-protocols
              libxkbcommon
              libdecor
              vulkan-loader
              llvmPackages.libclang
              linuxHeaders
            ];
            buildInputs = with pkgs; [
              wayland
              libdecor
              libxkbcommon
              vulkan-loader
            ];
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            C_INCLUDE_PATH = "${pkgs.linuxHeaders}/include:${pkgs.glibc.dev}/include";
          };

          # Build the workspace
          cargoArtifacts = craneLib.buildDepsOnly commonArgs;
          triad = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
          });
        in
        {
          default = triad;
          triad = triad;
        }
      );
    };
}
