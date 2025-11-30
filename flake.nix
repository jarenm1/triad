{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, flake-utils, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Pick whatever channel/version you want
        rustToolchain = fenix.packages.${system}.complete;           # stable + rust-src + clippy + rustfmt + llvm-tools + rust-analyzer
        # rustToolchain = fenix.packages.${system}.stable;            # smaller, also works
        # rustToolchain = fenix.packages.${system}.fromToolchainFile {
        #   file = ./rust-toolchain.toml;
        #   sha256 = "...";
        # };

      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain.toolchain
            pkgs.pkg-config
            pkgs.openssl
            pkgs.wayland
            pkgs.wayland-protocols
            pkgs.libxkbcommon
            pkgs.libdecor
            pkgs.vulkan-loader
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.wayland
            pkgs.libdecor
            pkgs.libxkbcommon
            pkgs.vulkan-loader
          ];
        };
      });
}
