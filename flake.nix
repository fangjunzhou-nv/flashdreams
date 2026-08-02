{
  description = "FlashDreams development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.aarch64-darwin
    ]
      (system:
        let
          inherit (nixpkgs) lib;
          pkgs = import nixpkgs { inherit system; };
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = with pkgs; [
              python312
              uv
            ];

            shellHook = ''
              # Always use the Nix-provided interpreter. Generic uv-managed
              # Python binaries do not run on NixOS without nix-ld.
              export UV_PYTHON_DOWNLOADS=never

              if [ -d .venv ]; then
                source .venv/bin/activate
                export PATH="$PWD/.venv/bin:$PATH"
              else
                echo "Environment not initialized. Run: uv sync --extra dev --extra runners"
              fi
            '';

            # Binary Python wheels need the C++ runtime, while CUDA loads the
            # host NVIDIA driver from NixOS's stable run path. Do not add the
            # full manylinux compatibility set here: LD_LIBRARY_PATH also
            # affects direnv's Nix subprocesses and can make them load an
            # incompatible libc or libstdc++.
            LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
              lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                "/run/opengl-driver"
              ]
            );
          };
        });
}
