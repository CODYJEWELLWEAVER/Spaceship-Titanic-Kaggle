{ pkgs ? import <nixpkgs> {}, config ? {} }:

with pkgs;

mkShell {
  buildInputs = [
    (python3.withPackages (py: with py; with python3Packages; [
      ipython
      jupyter
      numpy
      matplotlib
      tqdm
      pandas
      scikit-learn
    ]))
  ];
}