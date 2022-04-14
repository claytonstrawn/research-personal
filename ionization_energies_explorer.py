import numpy as np
import roman
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*4

#source = https://en.wikipedia.org/wiki/Molar_ionization_energies_of_the_elements
first10 = """
1	H	hydrogen	1312.0
2	He	helium	2372.3	5250.5
3	Li	lithium	520.2	7298.1	11,815.0
4	Be	beryllium	899.5	1757.1	14,848.7	21,006.6
5	B	boron	800.6	2427.1	3659.7	25,025.8	32,826.7
6	C	carbon	1086.5	2352.6	4620.5	6222.7	37,831	47,277.0
7	N	nitrogen	1402.3	2856	4578.1	7475.0	9444.9	53,266.6	64,360
8	O	oxygen	1313.9	3388.3	5300.5	7469.2	10,989.5	13,326.5	71,330	84,078.0
9	F	fluorine	1681.0	3374.2	6050.4	8407.7	11,022.7	15,164.1	17,868	92,038.1	106,434.3
10	Ne	neon	2080.7	3952.3	6122	9371	12,177	15,238.90	19,999.0	23,069.5	115,379.5	131,432
11	Na	sodium	495.8	4562	6910.3	9543	13,354	16,613	20,117	25,496	28,932	141,362
12	Mg	magnesium	737.7	1450.7	7732.7	10,542.5	13,630	18,020	21,711	25,661	31,653	35,458
13	Al	aluminium	577.5	1816.7	2744.8	11,577	14,842	18,379	23,326	27,465	31,853	38,473
14	Si	silicon	786.5	1577.1	3231.6	4355.5	16,091	19,805	23,780	29,287	33,878	38,726
"""
second10 = """
11	Na	sodium	159,076
12	Mg	magnesium	169,988	189,368
13	Al	aluminium	42,647	201,266	222,316
14	Si	silicon	45,962	50,502	235,196	257,923
"""

def get_energy_dict():
    convert_kJpermol_to_eV = 0.0103642688 #https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page)
    atoms = []
    energy_dict = {}
    energy_dict_cumulative = {}
    number_dict = {}
    for line in first10.split('\n'):
        if line in ['']:
            continue
        s = line.split('\t')
        number = s[0]
        atom = s[1]
        energies = s[3:]
        energies = [float(x.replace(',',''))*convert_kJpermol_to_eV for x in energies]
        energy_dict[atom] = energies
        number_dict[atom] = int(number)
        atoms+=[atom]
    for line in second10.split('\n'):
        if line == '':
            continue
        s = line.split('\t')
        atom = s[1]
        energies = s[3:]
        energies = [float(x.replace(',','')) for x in energies]
        energy_dict[atom] = energy_dict[atom]+energies
    for atom in atoms:
        energy_dict[atom] = np.array(energy_dict[atom])
        energy_dict_cumulative[atom] = np.cumsum(energy_dict[atom])
    ions = []
    for atom in atoms:
        max_ionization = number_dict[atom]+1
        for v in range(1,max_ionization+1):
            ions += [atom+' '+roman.toRoman(v)]
    ionization_energy_dict = {}
    for ion in ions:
        atom,ionization = ion.split(' ')
        ionization = roman.fromRoman(ionization)
        if ionization == 1:
            continue
        ionization_energy_dict[ion] = energy_dict_cumulative[atom][ionization-2]
    return atoms,ions,ionization_energy_dict

def get_cutoffs_dict(ions,redshift):
    #note: depends on quasarscan installation
    from quasarscan.utils.PI_field_defs import cutoffs_for_ion_at_redshift
    
    PI_cutoff_temps_dict = {}
    for ion in ions:
        try:
            PI_cutoff_temps_dict[ion] = np.median(cutoffs_for_ion_at_redshift(ion,redshift))
        except KeyError:
            PI_cutoff_temps_dict[ion] = np.nan
    return PI_cutoff_temps_dict

def get_energy_cutoffs_dicts(redshift=2.0):
    atoms,ions,ionization_energy_dict = get_energy_dict()
    return atoms,ions,ionization_energy_dict,get_cutoffs_dict(ions,redshift)

def view_energies(atoms,ions,ionization_energy_dict=None,cutoffs_dict = None):
    for ion in ions:
        atom,ionization = ion.split(' ')
        i = atoms.index(atom)
        ionization = roman.fromRoman(ionization)
        if ionization == 1:
            continue
        plt.semilogy(i,ionization_energy_dict[ion],'o',color = cycle_colors[i])
    ax = plt.gca()
    ax.set_xticks(range(len(atoms)))
    ax.set_xticklabels(atoms)
    ax.set_ylabel('ionization energy (eV)')
        
def plot_cutoffs_by_energies(atoms,ions,ionization_energy_dict,PI_cutoff_temps_dict,units = 'K',log = 'default'):
    used_atoms = []
    roman_symbols = []
    xs,ys = [],[]
    ax = plt.gca()
    if log == 'default':
        if units in ['K','eV']:
            func = ax.loglog
        elif units in ['relative']:
            func = ax.semilogx
    for ion in ions:
        if np.isnan(PI_cutoff_temps_dict[ion]):
            continue
        boltzmann = 8.617333262e-5 #eV/K
        if units == 'K':
            convert = 1
        elif units == 'eV':
            convert = 1.5*boltzmann
        elif units == 'relative':
            convert = 1.5*boltzmann/ionization_energy_dict[ion]
        xs+=[ionization_energy_dict[ion]]
        ys+=[PI_cutoff_temps_dict[ion]*convert]
        atom,ionization_r = ion.split(' ')
        used_atoms+=[] if atom in used_atoms else [atom]
        roman_symbols+=[]if ionization_r in roman_symbols else [ionization_r]
        i = used_atoms.index(atom)
        symbols = ['o','x','*','^','v','>','<','s','1','2','3','4','P','+','d','D']
        ionization = roman.fromRoman(ionization_r)
        if ionization == 1:
            continue
        func(ionization_energy_dict[ion],PI_cutoff_temps_dict[ion]*convert,
                   symbols[ionization-2],color = cycle_colors[i],markersize = 10)
    xlims = (1e1,4e3)
    ax.set_xlim(xlims[0],xlims[1])
    if units == 'K':
        ylims = (9e2,3e6)
        textloc = (1.5e1,7e5)
        ylabel = 'PI cutoff temperature (K)'
    elif units == 'eV':
        ylims = (1e-1,7e2)
        textloc = (1.5e1,1e2)
        ylabel = 'PI cutoff temperature (eV)'
    elif units == 'relative':
        ylims = (3e-2,1.5e-1)
        textloc = None
        ylabel = 'cutoff temp energy/particle\nover ionization energy'
    ax.set_ylim(ylims[0],ylims[1])
    ax.grid(which='both')
    ax.set_xlabel('ionization energy (eV)',size = 15)
    ax.set_ylabel(ylabel,size = 15)
    custom_lines1 = [Line2D([0], [0], marker = 'o',color = cycle_colors[j],linestyle = '') for j in range(len(used_atoms))]
    custom_lines2 = [Line2D([0], [0], marker = symbols[j],color = 'k',linestyle = '') for j in range(len(roman_symbols))]
    labels1 = used_atoms
    labels2 = roman_symbols
    legend1 = ax.legend(custom_lines1,labels1,bbox_to_anchor = (1.2,1.0),loc = 'upper right')
    ax.legend(custom_lines2,labels2,bbox_to_anchor = (1.36,1.0),loc = 'upper right')
    ax.add_artist(legend1)
    ax.tick_params(axis='both',labelsize=15)
    if units in ['K','eV']:
        xs = np.array(xs)
        ys = np.array(ys)
        logxs = np.log10(np.array(xs[np.logical_and(xs>xlims[0],xs<xlims[1])]))
        logys = np.log10(np.array(ys[np.logical_and(xs>xlims[0],xs<xlims[1])]))
        line = linregress(logxs, logys)
        slope,intercept,_,_,_ = line
        sortxs = np.array(sorted(logxs))
        linexs = 10**sortxs
        lineys = 10**(sortxs*slope+intercept)
        ax.text(textloc[0],textloc[1],'y = %.2f x ^ %.3f'%(10**intercept,slope),size= 20)
        func(linexs,lineys,'k:',linewidth = 3)
