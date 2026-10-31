{
  description = "vec-map-kit: vector tile maps as a native GPUI element";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # GPUI talks to Vulkan and the compositor directly, so the loader,
        # the Wayland/X11 client libraries and xkbcommon are needed at
        # runtime rather than through a toolkit.
        runtimeLibraries = with pkgs; [
          wayland
          vulkan-loader
          libxcb
          libxkbcommon
          fontconfig
          freetype
          libGL
        ];

        # A bare NixOS shell has no system font config; pin one containing
        # a UI font so GPUI's text system resolves a sans-serif family, plus
        # the families MapTiler / OpenMapTiles styles name in `text-font`.
        fontsConf = pkgs.makeFontsConf {
          fontDirectories = [
            (pkgs.ibm-plex.override { families = [ "sans" ]; })
            pkgs.open-sans
            pkgs.noto-fonts-lgc-plus
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer
          ];
          buildInputs = runtimeLibraries ++ [ pkgs.vulkan-headers ];

          env = {
            RUST_BACKTRACE = "1";
            FONTCONFIG_FILE = fontsConf;
            LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath runtimeLibraries}:/run/opengl-driver/lib";
          };

          # Vulkan ICD discovery for the vendor drivers NixOS exposes.
          shellHook = ''
            export XDG_DATA_DIRS="/run/opengl-driver/share:''${XDG_DATA_DIRS:-/usr/share}"
            # Patched GPUI crates referenced by Cargo.toml's [patch] section.
            ./script/vendor-gpui.sh
          '';
        };
      }
    );
}
