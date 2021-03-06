#!/usr/bin/env bash

# coloring
bold=$(tput bold)
green=$(tput setaf 2)
normal=$(tput sgr0)

echo "${bold}mypy${normal}"
mypy --ignore-missing-imports src/mlstarterpack tests \
    && echo "${green}OK${normal}" \
    || exit 1
echo

echo "${bold}tests${normal}"
py.test \
    && echo "${green}OK${normal}" \
    || exit 1
echo
