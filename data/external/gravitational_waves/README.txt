This Folder contains 117 distinct observations of Gravitational Waves. Each gravitaitonal Wave is generated by a binary system. The observable binary systems with the current technology are:
Neutron Star (NS) - Neutron Star (NS);
Neutron Star (NS) - Black Hole (BH);
Black Hole (BH) - Black Hole (BH).

The thirs binary system might be divided into multiple systems depending on the masses of the two components.

Each file Posterior_nXXX.csv contains the posterior of a given observation as provided by the LIGO/VIRGO consortium.
Posteriors are over a 4-dimensional space: m1 - m1 - s1 - s2. This is also the order of the columns in each file.

m1 and m2 are the masses of the primary and secondary components, respectively, of the binary system.
s1 adn s2 are the spins of the two components.


File "Labels.csv" contains the labels for each posterior. It is to be assumed that the first element of the file is the label for "Posterior_n001.csv" and so on.
Four classes have been estimated from the analysis:

NS - NS  (Label = 2)
NS - BH  (Label = 1)
Small BH - Small BH (Label = 3)
Large BH - Large BH (Label = 4)

If needed, another class can be provided (Small BH - Large BH), but this is to be agreed on.