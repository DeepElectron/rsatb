To improve the user experience, the user interface need to be as simple as possible. The steps for the user to conduct a calculation need to be minimized. The include both the steps for interfacing with other linear scaling DFT or tight-binding code, as well as the steps of the default format of rsatb itself.

Two step:
Interfacing:
1. geting other softwares output
2. rsatb -i xxx -o xxx
done!

This is the best for direct interfacing

One step:
internal API:
1. giving the structure, directly calls deeptb to generate the require input, and compute the properties in one step.