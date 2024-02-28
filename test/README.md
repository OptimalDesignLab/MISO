# Contents of the Test directory

The tests are built (from the `build` folder) by running `make tests`.  See the **Tests** Section of the main README file for more details.

If you only want to recompile and then run a single test, you can do so as follows:
```
make <test name without .cpp>.bin
mpirun -n 1 ./<test name without .cpp>.bin
```