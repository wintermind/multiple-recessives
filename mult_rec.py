# The objective of this simulation is to consider several scenarios for the management of multiple
# recessive alleles in a simulated population of dairy cattle. The basic idea is simple:
# + Each animal has parents, a sex code, a true breeding value for lifetime net merit, and a genotype
#   for the recessive alleles in the population;
# + Each recessive has a minor allele frequency in the base population and an economic value;
# + Matings will be based on parent averages, and at-risk matings will be penalized by the economic
#   value of each recessive.

# Import standard libraries
import copy
import itertools
import math
import subprocess
import sys
import random

# Import external libraries
import matplotlib
# Force matplotlib to not use any X-windows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist
import numpy as np
import numpy.ma as ma
from scipy.stats import bernoulli

# Setup the simulation
#    base_bulls:            Number of bulls in the base population (founders)
#    base_cows:             Number of cows in the base population (founders)
#    base_herds:            Number of herds in the population.
#    force_carriers:        If True, force at least one carrier of each sex.
#    force_best:            If True, force one carrier of each breed to have a
#                           TBV that is 4 SD above the mean for that sex.
#    recessives:            Dictionary of recessive alleles in the population.
#    check_tbv:             If True, plot histograms showing the distribution of
#                           the sire and dam TBV in the base population.
#    rng_key:                An integer used to see the random number generation. If
#                           False, a default value of 8675309 is used.
#    debug                  Flag to activate/deactivate debugging messages.


def setup(base_bulls=500, base_cows=2500, base_herds=100, force_carriers=True, force_best=True,
          recessives=[], check_tbv=False, rng_key=False, debug=True):

    # Base population parameters
    generation = 0                  # The simulation starts at generation 0. It's as though we're all C programmers.

    # Recessives are required since that's the whole point of this.
    if len(recessives) == 0:
        print '[setup]: The recessives dictionary passed to the setup() subroutine was empty! The program \
            cannot continue, and will halt.'
        sys.exit(1)

    # Seed the RNG
    try:
        np.random.seed(int(rng_key))
    except:
        np.random.seed(8675309)     # There was only one other option. Who can guess what it is?
        # If you want a random seed instead, you can change the previous line to:
        # np.random.seed()

    # The mean and standard deviation of the trait used to rank animals and make mating
    # decisions. The values here are for lifetime net merit in US dollars.
    mu_cows = 0.
    sigma = 200.

    # Create the base population of cows and bulls.

    # Assume bulls average 1.5 SD better than the cows.
    mu_bulls = mu_cows + (sigma * 1.5)

    # Make true breeding values
    base_cow_tbv = (sigma * np.random.randn(base_cows, 1)) + mu_cows
    base_bull_tbv = (sigma * np.random.randn(base_bulls, 1)) + mu_bulls

    # This dictionary will be used to store allele frequencies for each generation
    # of the simulation.
    freq_hist = {}
    freq_hist[0] = []
    for r in recessives:
        freq_hist[0].append(r[0])
     
    # Make the recessives. This is a little tricky. We know the minor
    # allele frequency in the base population, but we're going to simulate
    # the base population genotypes. We'll then calculate the allele
    # frequency in the next generation, which means that we need a table to
    # store the values over time. This is based on formulas presented in:
    #
    #     Van Doormaal, B.J., and G.J. Kistemaker. 2008. Managing genetic
    #     recessives in Canadian Holsteins. Interbull Bull. 38:70-74.
    #
    # On 6/2/14 I changed this so that there can be non-lethal recessives
    # (e;g;, polled) so that I can convince myself that the code is working
    # okay.

    # Initialize an array of zeros that will be filled with genotypes
    base_cow_gt = np.zeros((base_cows, len(recessives)))
    base_bull_gt = np.zeros((base_bulls, len(recessives)))
    for r in xrange(len(recessives)):
        # Get the MAF for the current recessive
        r_freq = recessives[r][0]
        # The recessive is a lethal
        if recessives[r][2] == 1:
            # Compute the frequency of the AA and Aa genotypes
            denom = (1. - r_freq)**2 + (2 * r_freq * (1. - r_freq))
            f_dom = (1. - r_freq)**2 / denom
            f_het = (2 * r_freq * (1. - r_freq)) / denom
            print 'This recessive is ***LETHAL***'
            print 'Recessive %s (%s), generation %s:' % (r, recessives[r][3], generation)
            print '\tp = %s' % (1. - r_freq)
            print '\tq = %s' % r_freq
            print '\tf(AA) = %s' % f_dom
            print '\tf(Aa) = %s' % f_het
            # Assign genotypes by drawing a random Bernoulli variate where the
            # parameter is the probability of an AA genotype. A value of 1 means
            # "AA", and a value of 0 means "Aa".
            for c in xrange(base_cows):
                base_cow_gt[c, r] = bernoulli.rvs(f_dom)
            for b in xrange(base_bulls):
                base_bull_gt[b, r] = bernoulli.rvs(f_dom)
            if force_carriers:
                # I want to force at least one carrier for each mutation so that the
                # vagaries of the RNG don't thwart me.
                base_cow_gt[r, r] = 0
                base_bull_gt[r, r] = 0
                print '\t[setup]: Forcing carriers to bend Nature to my will...'
                print '\t[setup]: \tCow %s is a carrier for recessive %s (%s)' % (r, recessives[r][3], r)
                print '\t[setup]: \tBull %s is a carrier for recessive %s (%s)' % (r, recessives[r][3], r)
        # The recessive is NOT lethal
        else:
            # Compute the frequency of the AA and Aa genotypes
            f_dom = (1. - r_freq)**2
            f_het = (2 * r_freq * (1. - r_freq))
            f_rec = r_freq**2
            print 'This recessive is ***NOT LETHAL***'
            print 'Recessive %s (%s), generation %s:' % (r, recessives[r][3], generation)
            print '\tp = %s' % (1. - r_freq)
            print '\tq = %s' % r_freq
            print '\tf(AA) = %s' % f_dom
            print '\tf(Aa) = %s' % f_het
            print '\tf(aa) = %s' % f_rec
            # Assign genotypes by drawing a random Bernoulli variate for each
            # parental allele. The parameter is the probability of an "A" allele.
            # A value of 1 assigned to the cow (bull) genotype means "AA", a
            # value of 0 means "Aa", and a value of -1 means "aa".
            for c in xrange(base_cows):
                # Get the cow's genotype -- since the parameter we're
                # using is the major allele frequency (p), a success (1) is
                # an "A" allele, and a failure is an "a" allele.
                s_allele = bernoulli.rvs(1. - r_freq)
                d_allele = bernoulli.rvs(1. - r_freq)
                if s_allele == 1 and d_allele == 1:
                    base_cow_gt[c, r] = 1
                elif s_allele == 0 and d_allele == 0:
                    base_cow_gt[c, r] = -1
                else:
                    base_cow_gt[c, r] = 0
            for b in xrange(base_bulls):
                # Get the bull's genotype -- since the parameter we're
                # using is the major allele frequency (p), a success (1) is
                # an "A" allele, and a failure is an "a" allele.
                s_allele = bernoulli.rvs(1. - r_freq)
                d_allele = bernoulli.rvs(1. - r_freq)
                if s_allele == 1 and d_allele == 1:
                    base_bull_gt[b, r] = 1
                elif s_allele == 0 and d_allele == 0:
                    base_bull_gt[b, r] = -1
                else:
                    base_bull_gt[b, r] = 0

            # You may want to force at least one carrier for each mutation so that the
            # vagaries of the RNG don't thwart you. If you don't do this, then your
            # base population may not have any minor alleles for a rare recessive.
            if force_carriers:
                base_cow_gt[r, r] = 0
                base_bull_gt[r, r] = 0
                print '\t[setup]: Forcing there to be a carrier for each recessive, i.e., bending Nature to my will.'
                print '\t[setup]: \tCow %s is a carrier for recessive %s (%s)' % (r, r, recessives[r][3])
                print '\t[setup]: \tBull %s is a carrier for recessive %s (%s)' % (r, r, recessives[r][3])
            
    # Storage
    cows = []                       # List of live cows in the population
    bulls = []                      # List of live bulls in the population
    dead_cows = []                  # List of dead cows in the population (history)
    dead_bulls = []                 # List of dead bulls in the population (history)
    id_list = []

    # Add animals to the bull list.
    for i in xrange(base_cows):
        # The list contents are:
        # animal ID, sire ID, dam ID, generation, sex, herd, alive/dead, reason dead, when dead, TBV, genotype
        # "generation" is the generation in which the base population animal was born, not its actual
        # age.
        c = i + 1
        if c in id_list:
            if debug:
                print '[setup]: Error! A cow with ID %s already exists in the ID list!' % c
        c_list = [c, 0, 0, (-1*random.randint(0, 4)), 'F', random.randint(0, base_herds), 'A',
                  '', -1, base_cow_tbv.item(i), []]
        for r in xrange(len(recessives)):
            c_list[-1].append(base_cow_gt.item(i, r))
        cows.append(c_list)
        id_list.append(c)

    # Add animals to the bull list.
    for i in xrange(base_bulls):
        b = i + 1 + base_cows
        if b in id_list:
            if debug:
                print '[setup]: Error! A bull with ID %s already exists in the ID list!' % b
        b_list = [b, 0, 0, (-1*random.randint(0, 9)), 'M', random.randint(0, base_herds), 'A', '',
                  -1, base_bull_tbv.item(i), []]
        for r in xrange(len(recessives)):
            b_list[-1].append(base_bull_gt.item(i,r))
        bulls.append(b_list)
        id_list.append(b)

    ### This worked fine in an IPython notebook, needs check here.
    if check_tbv:
        # Check the distribution of bull and cow TBV
        #min_data = np.r_[base_cow_tbv, base_bull_tbv].min()
        #max_data = np.r_[base_cow_tbv, base_bull_tbv].max()
        #print min_data, max_data
        hist(base_cow_tbv, normed=True, color="#6495ED", alpha=.5)
        hist(base_bull_tbv, normed=True, color="#F08080", alpha=.5)

    return cows, bulls, dead_cows, dead_bulls, freq_hist


# <markdowncell>

# Okay, now we've got at least a rough draft of the setup. Now we need to get code in place
# to simulate generation 1, which can then be generalized to *n* generations. In order to do
# this, we actually need to make a bunch of decisions. Here's an outline of what needs to happen
# each generation:
# 
# * The generation counter needs to be incremented
# * We need to mate cows and create their offspring, including genotypes
# * "Old" cows need to be culled so that the population size is maintained
# * Minor allele frequencies in the recessives lists need to be updated
#
# cows          : A list of live cow records
# bulls         : A list of live bull records
# dead_cows     : A list of dead cow records
# dead_bulls    : A list of dead bull records
# generation    : The current generation in the simulation
# recessives    : A Python list of recessives in the population
# max_matings   : The maximum number of matings permitted for each bull
# debug         : Flag to activate/deactivate debugging messages


def random_mating(cows, bulls, dead_cows, dead_bulls, generation, recessives, max_matings=50, debug=False):
    if max_matings <= 0:
        print "[random_mating]: max_matings cannot be <= 0! Setting to 50."
        max_matings = 50
    if max_matings * len(bulls) < len(cows):
        print "[random_mating]: You don't have enough matings to breed all cows1"
    # Make a list of bulls so that we can track the number of matings for each
    matings = {}
    new_cows = []
    new_bulls = []
    for b in bulls:
        matings[b[0]] = 0
    next_id = len(cows) + len(bulls) + len(dead_cows) + len(dead_bulls) + 1
    # Now we need to randomly assign mates. We do this as follows:
    #     1. Loop over cow list
    if debug:
        print '%s bulls in list for mating' % len(bulls)
        print '%s cows in list for mating' % len(cows)
    for c in cows:
        # Is the cow alive?
        if c[6] == 'A':
            cow_id = c[0]
            mated = False
            if debug:
                print 'Mating cow %s' % cow_id
            while not mated:
    #     2. For each cow, pick a bull at random
                bull_to_use = random.randint(0, len(bulls)-1)
                bull_id = bulls[bull_to_use][0]
                if debug:
                    print 'Using bull %s (ID %s)' % (bull_to_use, bull_id)
    #     3. If the bull is alive and has matings left then use him            
                if bulls[bull_to_use][6] == 'A' and matings[bull_id] < max_matings:
                    if debug:
                        print 'bull %s (ID %s) is alive and has available matings' % (bull_to_use, bull_id)
                    # Create the resulting calf
                    calf = create_new_calf(bulls[bull_to_use], c, recessives, next_id, generation, debug=debug)
                    if debug:
                        print calf
                    if calf[4] == 'F':
                        new_cows.append(calf)
                    else:
                        new_bulls.append(calf)
    #         Done!
                    next_id += 1
                    mated = True
                else:
                    if debug:
                        print 'bull %s (ID %s) is not alive or does not have available matings' % (bull_to_use, bull_id)
    for nc in new_cows:
        if nc[6] == 'A':
            cows.append(nc)
        else:
            dead_cows.append(nc)
    for nb in new_bulls:
        if nb[6] == 'A':
            bulls.append(nb)
        else:
            dead_bulls.append(nb)
    return cows, bulls, dead_cows, dead_bulls

# Okay, now we've got at least a rough draft of the setup. Now we need to get
# code in place to simulate generation 1, which can then be generalized to *n*
# generations. In order to do this, we actually need to make a bunch of
# decisions. Here's an outline of what needs to happen each generation:
# 
# * The generation counter needs to be incremented
# * We need to make a list of the top "pct" bulls, and there is no limit
#   to the number of matings for each bull, so we will mate randomly from
#   within the top group.
# * We need to mate cows and create their offspring, including genotypes
# * "Old" cows need to be culled so that the population size is maintained
# * Minor allele frequencies in the recessives lists need to be updated
#
# cows          : A list of live cow records
# bulls         : A list of live bull records
# dead_cows     : A list of dead cow records
# dead_bulls    : A list of dead bull records
# generation    : The current generation in the simulation
# recessives    : A Python list of recessives in the population
# pct           : The proportion of bulls to retain for mating
# debug         : Flag to activate/deactivate debugging messages


def toppct_mating(cows, bulls, dead_cows, dead_bulls, generation,
                  recessives, pct=0.10, debug=False):
    if debug:
        print '[toppct_mating]: PARMS:\n\tgeneration: %s\n\trecessives; %s\n\tpct: %s\n\tdebug: %s' % \
            (generation, recessives, pct, debug)
    # Never trust users, they are lying liars
    if pct < 0.0 or pct > 1.0:
        print '[toppct_mating]: %s is outside of the range 0.0 <= pct <= 1.0, changing to 0.10' % ( pct )
        pct = 0.10
    # Sort bulls on TBV in ascending order
    bulls.sort(key=lambda x: x[8])
    # How many do we keep?
    b2k = int(pct*len(bulls))
    if debug:
        print '[toppct_mating]: Using %s bulls for mating' % b2k
    # Set-up data structures
    new_cows = []
    new_bulls = []
    next_id = len(cows) + len(bulls) + len(dead_cows) + len(dead_bulls) + 1
    # Now we need to randomly assign mates. We do this as follows:
    #     1. Loop over cow list
    if debug: 
        print '\t[toppct_mating]: %s bulls in list for mating' % len(bulls)
        print '\t[toppct_mating]: %s cows in list for mating' % len(cows)
    for c in cows:
        # Is the cow alive?
        if c[6] == 'A':
            cow_id = c[0]
            mated = False
            if debug:
                print '\t[toppct_mating]: Mating cow %s' % cow_id
            while not mated:
    #     2. For each cow, pick a bull at random
                # Note the offset index to account for the fact that we're picking only
                # from the top pct of the bulls.
                bull_to_use = random.randint(len(bulls)-b2k, len(bulls)-1)
                bull_id = bulls[bull_to_use][0]
                #if debug: print 'Using bull %s (ID %s)' % ( bull_to_use, bull_id )
    #     3. If the bull is alive then use him            
                if bulls[bull_to_use][6] == 'A':
                    if debug:
                        print 'bull %s (ID %s) is alive' % (bull_to_use, bull_id)
                    # Create the resulting calf
                    calf = create_new_calf(bulls[bull_to_use], c, recessives, next_id, generation, debug=debug)
                    if debug:
                        print calf
                    if calf[4] == 'F':
                        new_cows.append(calf)
                    else:
                        new_bulls.append(calf)
    #   Done!
                    next_id += 1
                    mated = True
                else:
                    if debug:
                        print '[toppct_mating]: bull %s (ID %s) is not alive' % (bull_to_use, bull_id)
    if debug:
        print '\t[toppct_mating]: %s animals in original cow list' % len(cows)
        print '\t[toppct_mating]: %s animals in new cow list' % len(new_cows)
        print '\t[toppct_mating]: %s animals in original bull list' % len(bulls)
        print '\t[toppct_mating]: %s animals in new bull list' % len(new_bulls)
    for nc in new_cows:
        if nc[6] == 'A':
            cows.append(nc)
        else:
            dead_cows.append(nc)
    for nb in new_bulls:
        if nb[6] == 'A':
            bulls.append(nb)
        else:
            dead_bulls.append(nb)
    if debug:
        print '\t[toppct_mating]: %s animals in final cow list' % len(cows)
        print '\t[toppct_mating]: %s animals in final bull list' % len(bulls)
    return cows, bulls, dead_cows, dead_bulls


# This function returns the largest animal ID in the population + 1, which
# is used as the starting ID for the next generation of calves.


def get_next_id(cows, bulls, dead_cows, dead_bulls):
    id_list = []
    for c in cows:
        id_list.append(int(c[0]))
    for dc in dead_cows:
        id_list.append(int(dc[0]))
    for b in bulls:
        id_list.append(int(b[0]))
    for db in dead_bulls:
        id_list.append(int(db[0]))
    id_list.sort()
    next_id = id_list[-1] + 1
    return next_id


# This routine uses an approach similar to that of Pryce et al. (2012) allocate matings of bulls
# to cows. Parent averages are discounted for any increase in inbreeding in the progeny, and
# they are further discounted to account for the effect of recessives on lifetime income.
#
# cows          : A list of live cow records
# bulls         : A list of live bull records
# dead_cows     : A list of dead cow records
# dead_bulls    : A list of dead bull records
# generation    : The current generation in the simulation
# recessives    : A Python list of recessives in the population
# max_matings   : The maximum number of matings permitted for each bull
# base_herds    : Number of herds in the population.
# debug         : Flag to activate/deactivate debugging messages


def pryce_mating(cows, bulls, dead_cows, dead_bulls, generation,
                 recessives, max_matings=500, base_herds=100, debug=False):

    if debug:
        print '[pryce_mating]: PARMS:\n\tgeneration: %s\n\tmax_matings: %s\n\tbase_herds: %s\n\tdebug: %s' % \
            (generation, max_matings, base_herds, debug)
        for r in recessives:
            print '\t%s' % r
    # Never trust users, they are lying liars
    if max_matings < 0:
        print '[pryce_mating]: %s is less than 0, changing num_matings to 500.' % max_matings
        max_matings = 500
    if not type(max_matings) is int:
        print '[pryce_mating]: % is not not an integer, changing num_matings to 500.' % max_matings
    #
    # Now, we're going to need to construct a pedigree that includes matings of all cows in
    # each herd to the bulls randomly assigned to that herd. Bulls are randomly assigned to
    # herds to reflect different sire selection policies. It is faster to calculate the
    # inbreeding of the potential offspring than it is to calculate relationships among parents
    # because the latter requires that we store relationships among all parents.
    #
    #next_id = len(cows) + len(bulls) + len(dead_cows) + len(dead_bulls) + 1
    next_id = get_next_id(cows, bulls, dead_cows, dead_bulls)
    if debug:
        print '[pryce_mating]: next_id = %s in generation %s' % (next_id, generation)
    matings = {}
    pedigree = []
    id_list = []
    for c in cows:
        new_cow = '%s %s %s %s\n' % (c[0], c[1], c[2], c[3]+10)         # INBUPGF90 doesn't like negative birth years.
        pedigree.append(new_cow)
        if c[0] not in id_list:
            id_list.append(c[0])
    for dc in dead_cows:
        dead_cow = '%s %s %s %s\n' % (dc[0], dc[1], dc[2], dc[3]+10)
        pedigree.append(dead_cow)
        if dc[0] not in id_list:
            id_list.append(dc[0])
    for b in bulls:
        new_bull = '%s %s %s %s\n' % (b[0], b[1], b[2], b[3]+10)
        pedigree.append(new_bull)
        if b[0] not in id_list:
            id_list.append(b[0])
        matings[b[0]] = 0
    for db in dead_bulls:
        dead_bull = '%s %s %s %s\n' % (db[0], db[1], db[2], db[3]+10)
        pedigree.append(dead_bull)
        if db[0] not in id_list:
            id_list.append(db[0])
    if debug:
        print '[pryce_mating]: %s "old" animals in pedigree in generation %s' % \
            (len(pedigree), generation)
    # Now we need to create faux offspring of the living bulls and cows because it is faster to
    # compute inbreeding than relationships.
    calfcount = 0
    n_bulls = int(round(len(bulls) / base_herds, 0))
    if debug:
        print '[pryce_mating]: base_herds: %s' % base_herds
        print '[pryce_mating]: len(bulls): %s' % len(bulls)
        print '[pryce_mating]: len(bulls) / base_herds: %s' % (float(len(bulls)) / float(base_herds))
        print '[pryce_mating]: round(len(bulls) / base_herds, 0): %s' % (round(len(bulls) / base_herds, 0))
        print '[pryce_mating]: int(round(len(bulls) / base_herds, 0)): %s' % (int(round(len(bulls) / base_herds, 0)))
        print '[pryce_mating]: n_bulls: %s' % n_bulls
    for herd in xrange(base_herds):
        random.shuffle(bulls)               # Randomly assign bulls to cows
        herd_bulls = bulls[0:n_bulls+1]
        herd_cows = [c for c in cows if c[5] == herd]
        if debug:
            print '[pryce_mating]: herd_bulls:\n\n%s' % herd_bulls
        for b in herd_bulls:
            for c in herd_cows:
                calf_id = str(b[0])+'00'+str(c[0])
                if calf_id in id_list:
                    if debug:
                        print '\t[pryce_mating]: Error! A calf with ID %s already exists in the ID list in \
                            generation %s!' % (calf_id, generation)
                new_calf = '%s %s %s %s\n' % (calf_id, b[0], c[0], generation+10)
                pedigree.append(new_calf)
                calfcount += 1
                #next_id += 1
    if debug:
        print '[pryce_mating]: %s calves added to pedigree in generation %s' % \
            (calfcount, generation)
        print '[pryce_mating]: %s total animals in pedigree in generation %s' % \
            (len(pedigree), generation)
    # Write the pedigree to a file.
    pedfile = 'pedigree_%s.txt' % generation
    if debug:
        print '[pryce_mating]: Writing pedigree to %s' % pedfile
    ofh = file(pedfile, 'w')
    for p in pedigree:
        ofh.write(p)
    ofh.close()
    # PyPedal is just too slow when the pedigrees are large (e.g., millons of records), so
    # we're going to use Ignacio Aguilar's INBUPGF90 program.
    logfile = 'pedigree_%s.log' % generation
    #Several methods can be used:
    # 1 - recursive as in Aguilar & Misztal, 2008 (default)
    # 2 - recursive but with coefficients store in memory, faster with large number of
    #     generations but more memory requirements
    # 3 - method as in Meuwissen & Luo 1992
    callinbupgf90 = ['inbupgf90', '--pedfile', pedfile, '--method', '3', '--yob', '>', logfile, '2>&1&']
    if debug:
        print '[pryce_mating]: Calling inbupgf90 to calculate COI:\n\t%s' % callinbupgf90
    subprocess.call(callinbupgf90, shell=False)
    #print '[pryce_mating]: inbupgf90 returned code ',  p.stdout.read()

    # Load the COI into a dictionary keyed by original animal ID
    coifile = 'pedigree_%s.txt.solinb' % generation
    if debug:
        print '[pryce_mating]: Putting coefficients of inbreeding from %s.solinb in a dictionary' \
            % pedfile
    inbr = {}
    ifh = open(coifile, 'r')
    for line in ifh:
        pieces = line.split()
        inbr[pieces[0]] = float(pieces[1])
    ifh.close()

    # Setup the B_0 matrix, which will contain PA BV plus an inbreeding penalty
    flambda = 12.           # Loss of NM$ per 1% increase in inbreeding
                            # Smith L.A., Cassell B., Pearson R.E. (1998) The effects of
                            # inbreeding on the lifetime performance of dairy cattle.
                            # J. Dairy Sci., 81, 2729--2737.
    #b_mat = np.zeros((len(bulls), len(cows)))
    #f_mat = np.zeros((len(bulls), len(cows)))
    b_mat = ma.zeros((len(bulls), len(cows)))
    f_mat = ma.zeros((len(bulls), len(cows)))
    if debug:
        print '[pryce_mating]: Populating B_0'
    bids = [b[0] for b in bulls]
    cids = [c[0] for c in cows]
    for b in bulls:
        bidx = bids.index(b[0])
    for c in cows:
        cidx = cids.index(c[0])
        calf_id = str(b[0])+'00'+str(c[0])
        # If there is a hit on this key, then it means that the bull was
        # used in the herd in which the cow lives.
        try:
            # Update the matrix of inbreeding coefficients.
            f_mat[bidx, cidx] = inbr[calf_id]
            # Now, we need to adjust the parent averages to account for
            # inbreeding effects, and the economic impacts of the recessives.
            b_mat[bidx, cidx] = (0.5 * (b[8] + c[8])) - \
                (inbr[calf_id] * 100 * flambda)
            for r in xrange(len(recessives)):
                # What are the parent genotypes?
                b_gt = b[-1][r]
                c_gt = c[-1][r]
                if b_gt == -1 and c_gt == -1:           # aa genotypes
                    # The calf will definitely be affected, so we adjust
                    # the parent average by the full "value" of an aa
                    # calf.
                    b_mat[bidx, cidx] -= recessives[r][1]
                elif b_gt == 1 and c_gt == 1:           # AA genotypes
                    # The calf cannot be aa, so there is no adjustment to
                    # the parent average.
                    pass
                else:
                    # There is a 1/4 chance of having an affected calf,
                    # so the PA is adjusted by 1/4 of the "value" of an
                    # aa calf.
                    b_mat[bidx, cidx] -= (0.25 * recessives[r][1])
        # If there is not matching key in the dictionary, then that
        # cow-bull combination was not evaluated, which means the bull
        # was not used in the same herd as the cow. We're setting those
        # cells to "NA" instead of a proxy value, e.g., -999, since
        # NumPy functions and methods are supposed to understand them.
        except KeyError:
            #b_mat[bidx, cidx] = NA
            b_mat[bidx, cidx] = ma.masked
            #f_mat[bidx, cidx] = NA
            f_mat[bidx, cidx] = ma.masked

    # Now we're going to actually allocate mate pairs
    #m_mat = np.zeros((len(bulls), len(cows)))
    m_mat = ma.zeros((len(bulls), len(cows)))
    # Loop over columns (cows) to allocate the best mate

    # From Pryce et al. (2012) (http://www.journalofdairyscience.org/article/S0022-0302(11)00709-0/fulltext#sec0030)
    # A matrix of selected mates (mate allocation matrix; M) was
    # constructed, where Mij=1 if the corresponding element, Bij
    # was the highest value in the column Bj; that is, the maximum
    # value of all feasible matings for dam j, all other elements
    # were set to 0, and were rejected sire and dam combinations.
    # Sort bulls on ID in ascending order
    bulls.sort(key=lambda x: x[0])
    #bull_id_list = [b[0] for b in bulls]
    cow_id_list = [c[0] for c in cows]
    new_bulls = []
    new_cows = []
    if debug:
        print '\t[pryce_mating]: Beginning loop to assign mates and update M_0'
    for c in cows:
        if c[6] == 'A':
            # What column in b_mat corresponds to cow c?
            cow_loc = cow_id_list.index(c[0])
            # Get a vector of indices that would result in a sorted list.
            sorted_bulls = ma.argsort(b_mat[:, cow_loc])
            if debug:
                print '\t[pryce_mating]: sorted_bulls has %s animals' % len(sorted_bulls)
            # The first element in sorted_sires is the index of
            # the smallest element in b_mat[:,cow_loc]. The
            # last element in sorted_sires is the index of the
            # largest element in b_mat[:,cow_loc].
            for bidx in xrange(len(sorted_bulls)-1, -1, -1):
                # Does this bull still have matings available?
                #if debug:
                #    print '\t\t[pryce_mating]: checking bull with index %s' % bidx
                #    print '\t\t[pryce_mating]: sorted_bulls: %s' % sorted_bulls[bidx]
                #    print '\t\t[pryce_mating]: bulls       : %s' % bulls[sorted_bulls[bidx]]
                if matings[bulls[sorted_bulls[bidx]][0]] >= max_matings:
                    #if debug:
                    #    print '\t\t[pryce_mating]: Bull %s has no matings left, %s already used!' % \
                    #          (bulls[sorted_bulls[bidx]][0], matings[bulls[sorted_bulls[bidx]][0]])
                    pass
                elif bulls[sorted_bulls[bidx]][6] != 'A':
                    #if debug:
                    #    print '\t\t[pryce_mating]: Bull %s is dead!' % bulls[sorted_bulls[bidx]][0]
                    pass
                else:
                    # Allocate the mating
                    #if debug:
                    #    print '\t\t[pryce_mating]: sorted_bulls[%s] = %s' % (bidx, sorted_bulls[bidx])
                    #    print '\t\t[pryce_mating]: cow_lox = %s' % cow_loc
                    #    print '\t\t[pryce_mating]: m_mat[%s, %s] = %s' % (sorted_bulls[bidx], \
                    #        cow_loc, m_mat[sorted_bulls[bidx], cow_loc])
                    m_mat[sorted_bulls[bidx], cow_loc] = 1
                    # Increment the matings counter
                    matings[bulls[sorted_bulls[bidx]][0]] += 1
                    # Create the resulting calf
                    calf = create_new_calf(bulls[sorted_bulls[bidx]], c, recessives, next_id, generation, debug=debug)
                    # if debug: print calf
                    if calf[4] == 'F':
                        new_cows.append(calf)
                    else:
                        new_bulls.append(calf)
                    next_id += 1
                    # ...and, we're done.
                    break

    if debug:
        print '\t[pryce_mating]: %s animals in original cow list' % len(cows)
        print '\t[pryce_mating]: %s animals in new cow list' % len(new_cows)
        print '\t[pryce_mating]: %s animals in original bull list' % len(bulls)
        print '\t[pryce_mating]: %s animals in new bull list' % len(new_bulls)
    for nc in new_cows:
        if nc[6] == 'A':
            cows.append(nc)
        else:
            dead_cows.append(nc)
    for nb in new_bulls:
        if nb[6] == 'A':
            bulls.append(nb)
        else:
            dead_bulls.append(nb)
    if debug:
        print '\t[pryce_mating]: %s animals in final live cow list' % len(cows)
        print '\t[pryce_mating]: %s animals in final dead cow list' % len(dead_cows)
        print '\t[pryce_mating]: %s animals in final live bull list' % len(bulls)
        print '\t[pryce_mating]: %s animals in final dead bull list' % len(dead_bulls)

    # if debug:
    #     # The D_mat and F_mat matrices are used to compute summary statistics
    #     # for the offspring PTA and coefficients of inbreeding.
    #     #d_mat = ma.zeros(())
    #     d_mat = b_mat * m_mat
    #     f_mat = f_mat * m_mat
    #
    #     print 'b_mat: %s' % b_mat
    #     print 'd_mat: %s' % d_mat
    #     print 'f_mat: %s' % f_mat
    #     print 'm_mat: %s' % m_mat
    #
    #     # We might need ", skipna=True".
    #     print
    #     print '\t[pryce_mating]: Average PTA in D_mat            : ', ma.average(d_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Std. dev. of PTA in D_mat       : ', ma.std(d_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Minimum PTA in D_mat            : ', ma.min(d_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Maximum PTA in D_mat            : ', ma.max(d_mat.sum(axis=0))
    #     print
    #     print '\t[pryce_mating]: Average inbreeding in F_mat     : ', ma.average(f_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Std. dev. of inbreeding in F_mat: ', ma.std(f_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Minimum inbreeding in F_mat     : ', ma.min(f_mat.sum(axis=0))
    #     print '\t[pryce_mating]: Maximum inbreeding in F_mat     : ', ma.max(f_mat.sum(axis=0))

    return cows, bulls, dead_cows, dead_bulls


# I finally had to refactor the create-a-calf code into its own subroutine. This function
# returns a new animal record.
#
# sire          : The sire's ID
# dam           : The dam's ID
# recessives    : A Python list of recessives in the population
# calf_id       : The ID to be assigned to the new animal
# generation    : The current generation in the simulation
# debug         : Flag to activate/deactivate debugging messages


def create_new_calf(sire, dam, recessives, calf_id, generation, debug=False):
    # Is it a bull or a heifer?
    if bernoulli.rvs(0.50):
        sex = 'M'
    else:
        sex = 'F'
    # Compute the parent average
    tbv = (sire[9] + dam[9]) * 0.5
    # Add a Mendelian sampling term -- in this case, it's set to
    # +/- 1 genetic SD.
    tbv += (random.uniform(-1., 1.) * 200)
    # Form the animal record. Note that heifers are born into the same herd as their
    # dam.
    calf = [calf_id, sire[0], dam[0], generation, sex, dam[5], 'A', '', -1, tbv, []]
    # Check the bull and cow genotypes to see if the mating is at-risk
    # If it is, then reduce the parent average by the value of the recessive.
    c_gt = dam[-1]
    b_gt = sire[-1]
    for r in xrange(len(recessives)):
        # The simplest way to do this is to draw a gamete from each parent and
        # construct the calf's genotype from there. In the recessives array, a
        # 1 indicates an AA, 0 is an Aa, and a -1 is aa.
        #
        # Draw an allele from the sire -- a 0 is an "A", and a 1 is an "a".
        if b_gt[r] == 1:                       # AA genotype
            s_allele = 'A'
        elif b_gt[r] == 0:                     # Aa genotype
            s_allele = bernoulli.rvs(0.5)
            if s_allele == 0:
                s_allele = 'A'
            else:
                s_allele = 'a'
        else:                                  # aa genotype
            s_allele = 'a'
        # Draw an allele from the dam -- a 0 is an "A", and a 1 is an "a".
        if c_gt[r] == 1:                       # AA genotype
            d_allele = 'A'
        elif c_gt[r] == 0:                     # Aa genotype
            d_allele = bernoulli.rvs(0.5)
            if d_allele == 0:
                d_allele = 'A'
            else:
                d_allele = 'a'
        else:                                  # aa genotype
            d_allele = 'a'
        # Now, we construct genotypes.
        #
        # This mating produces only 'aa' genotypes.
        if s_allele == 'a' and d_allele == 'a':
            # The recessive is lethal.
            if recessives[r][2] == 1: 
                calf[6] = 'D'            # The calf is dead
                calf[7] = 'R'            # Because of a recessive lethal
                calf[8] = generation     # In utero
            # In either case (lethal or non-lethal) the genotype is the same.
            calf[-1].append(-1)
        # This mating produces only 'AA' genotype.
        elif s_allele == 'A' and d_allele == 'A':
            # But, oh noes, spontaneous mutation can ruin all teh DNA!!!
            # I put this in to try and keep the lethals from disappearing
            # from the population too quickly. That's why genotypes only
            # change from AA to Aa.
            if random.randint(1, 100001) == 1:
                if debug:
                    print '\t[create_new_calf]: A mutation in recessive %s (%s) happened when ' \
                        'bull %s was mated to cow %s to produce animal %s!' % (r, recessives[r][3],
                        sire[0], dam[0], calf_id)
                calf[-1].append(0)
            else:
                calf[-1].append(1)
        # These matings can produce only "Aa" genotypes.
        else:
            calf[-1].append(0)
    return calf


# This routine culls bulls each generation. The rules used are:
# 1.  Bulls cannot be more than 10 years old
# 2.  After that, bulls are sorted on PTA and only the top
#     max_bulls animals are retained.
#
# bulls         : A list of live bull records
# dead_bulls    : A list of dead bull records
# generation    : The current generation in the simulation
# max_bulls     : The maximum number of bulls that can be alive at one time
# debug         : Flag to activate/deactivate debugging messages


def cull_bulls(bulls, dead_bulls, generation, max_bulls=250, debug=False):
    if debug:
        print '[cull_bulls]: live bulls: %s' % len(bulls)
        print '[cull_cows]: dead bulls: %s' % len(dead_bulls)
    if max_bulls <= 0:
        print "[cull_bulls]: max_bulls cannot be <= 0! Setting to 250."
        max_bulls = 250
    if debug:
        age_distn(bulls, generation)
    # This is the age cull
    n_culled = 0
    for b in bulls:
        if (generation - b[3]) > 10:
            b[6] = 'D'            # This bull is dead
            b[7] = 'A'            # From age
            b[8] = generation     # In the current generation
            dead_bulls.append(b)  # Add it to the dead bulls list
            n_culled += 1
    if debug:
        print '\t[cull_bulls]: %s bulls culled for age in generation %s (age>10)' % (n_culled, generation)
    # Now we have to remove the dead bulls from the bulls list
    bulls[:] = [b for b in bulls if b[6] == 'A']
    # Now we're going to sort on TBV
    bulls.sort(key=lambda x: x[9])
    # Check to see if we need to cull on number (count).
    if len(bulls) <= max_bulls:
        if debug:
            print '\t[cull_bulls]: No bulls culled in generation %s (bulls<max_bulls)' % generation
        return bulls, dead_bulls
    # If this culling is necessary then we need to update the records of the
    # culled bulls and move them into the dead_bulls list. We cull bulls at
    # random.
    else:
        n_culled = 0
        random.shuffle(bulls)
        for b in bulls[0:len(bulls)-max_bulls-1]:
            b[6] = 'D'           # This bull is dead
            b[7] = 'N'           # Because there were too many of them
            b[8] = generation    # In the current generation
            dead_bulls.append(b)
            n_culled += 1
        bulls = bulls[len(bulls)-max_bulls:]
        if debug:
            print '\t[cull_bulls]: %s bulls culled because of excess population in generation %s ' \
                  '(bulls>max_bulls)' % (n_culled, generation)
        return bulls, dead_bulls

# Print a table showing how many animals of each age are in the population. Returns a
# dictionary of results. If the "show" parameter is True then print the table to
# the console.
#
# animals       : A list of live animal records
# generation    : The current generation in the simulation
# show          : Flag to activate/deactivate printing of the age distribution


def age_distn(animals, generation, show=True):
    ages = {}
    for a in animals:
        age = generation - a[3]
        if not ages.has_key(age):
            ages[age] = 0
        ages[age] += 1
    if show:
        keys = ages.keys()
        keys.sort()
        print '\tAnimal age distribution'
        for k in keys:
            print '\t%s:\t\t%s' % (k, ages[k])
    return ages


# This routine culls cows each generation. The rules used are:
# 1.  Cows cannot be more than 5 years old
# 2.  There is an [optional] involuntary cull at a user-specified rate 
# 3.  After that, cows are culled at random to get down to the maximum herd size
#
# cows          : A list of live cow records
# dead_cows     : A list of dead cow records
# generation    : The current generation in the simulation
# max_cows      : The maximum number of cows that can be alive at one time
# culling_rate  : The proportion of cows culled involuntarily each generation
# debug         : Flag to activate/deactivate debugging messages


def cull_cows(cows, dead_cows, generation, max_cows=0, culling_rate=0.0, debug=False):
    if debug:
        print '[cull_cows]: live cows: %s' % len(cows)
        print '[cull_cows]: dead cows: %s' % len(dead_cows)
    # 0 means keep all cows after age-related and involuntary culling
    if max_cows < 0:
        print "[cull_cows]: max_cows cannot be < 0! Setting to 0."
        max_cows = 0
    # This is the age cull
    n_culled = 0
    for c in cows:
        if (generation - c[3]) > 5:
            c[6] = 'D'            # This cow is dead
            c[7] = 'A'            # Because of her age
            c[8] = generation     # In the current generation
            dead_cows.append(c)   # Add it to the dead cows list
            n_culled += 1
    if debug: print '\t[cull_cows]: %s cows culled for age in generation %s' % (n_culled, generation)
    # Now we have to remove the dead animals from the cows list
    cows[:] = [c for c in cows if c[6] == 'A']
    # Now for the involuntary culling!
    if culling_rate > 0:
        n_culled = 0
        for c in cows:
            if random.uniform(0, 1) < culling_rate:
                c[6] = 'D'             # This cow is dead
                c[7] = 'C'             # Because of involuntary culling
                c[8] = generation      # In the current generation
                dead_cows.append(c)    # Add it to the dead cows list
                n_culled += 1
        if debug:
            print '\t[cull_cows]: %s cows involuntarily culled in generation %s' % (n_culled, generation)
    # Now we have to remove the dead animals from the cows list
    cows[:] = [c for c in cows if c[6] == 'A']
    # Now we're going to sort on TBV in ascending order
    #cows.sort(key=lambda x: x[9])
    # Instead of culling from only the low tail, we'll cull at random.
    random.shuffle(cows)
    # Check to see if we need to cull on number (count).
    if max_cows == 0:
        if debug:
            print '\t[cull_cows]: No cows were culled to maintain herd size in generation %s (max_cows=0)' % generation
        return cows, dead_cows
    elif len(cows) < max_cows:
        if debug:
            print '\t[cull_cows]: No cows were culled to maintain herd size in generation %s (n<=max_cows)' % generation
        return cows, dead_cows
    # If this culling is necessary then we need to update the records of the
    # culled bulls and move them into the dead_bulls list.
    else:
        c_diff = len(cows) - max_cows
        for c in cows[0:c_diff-1]:
            c[6] = 'D'           # This cow is dead
            c[7] = 'N'           # Because there were too many of them
            c[8] = generation    # In the current generation
            dead_cows.append(c)
        cows = cows[c_diff:]
        if debug: print '\t[cull_cows]: %s cows were culled to maintain herd size in generation %s (cows>max_cows)'\
                        % (c_diff, generation)
        return cows, dead_cows


# Compute simple summary statistics of TBV for the list of animals passed in:
#    sample mean
#    min, max, and count
#    sample variance and standard deviation


def animal_summary(animals):
    total = 0.
    count = 0.
    tmin = float('inf')
    tmax = float('-inf')
    sumx = 0.
    sumsq = 0.
    for a in animals:
        count += 1
        total = total + a[9]
        if a[9] < tmin:
            tmin = a[9]
        if a[9] > tmax:
            tmax = a[9]
        sumx = sumx + a[9]
        sumsq += a[9]**2
    if count == 0.:
        samplemean = -999.
        samplevar = -999.
        samplestd = -999.
    else:
        samplemean = total / count
        samplevar = (1 / (count-1)) * (sumsq - (sumx**2 / count))
        samplestd = math.sqrt(samplevar)
    return count, tmin, tmax, samplemean, samplevar, samplestd


# The easy way to determine the current MAF for each recessive is to count
# the number of copies of each "a" allele in the current population of live
# animals.


def update_maf(cows, bulls, generation, recessives, freq_hist):
    minor_allele_counts = []
    for r in recessives:
        minor_allele_counts.append(0)
    # Loop over the bulls list and count
    for b in bulls:
        for r in xrange(len(recessives)):
            # A genotype code of 0 is a heterozygote (Aa), and a 1 is a homozygote (AA)
            if b[-1][r] == 0:
                minor_allele_counts[r] += 1
            # As of 06/02/2014 homozygous recessives aren't necessarily lethal, so
            # we need to make sure that we count them, too.
            if b[-1][r] == -1:
                minor_allele_counts[r] += 2
    # Loop over the cows list and count
    for c in cows:
        for r in xrange(len(recessives)):
            # A genotype code of 0 is a heterozygote (Aa), and a 1 is a homozygote (AA)
            if c[-1][r] == 0:
                minor_allele_counts[r] += 1
            # As of 06/02/2014 homozygous recessives aren't necessarily lethal, so
            # we need to make sure that we count them, too.
            if c[-1][r] == -1:
                minor_allele_counts[r] += 2
    # Now we have to calculate the MAF for each recessive
    total_alleles = 2 * (len(cows) + len(bulls))
    freq_hist[generation] = []
    for r in xrange(len(recessives)):
        # r_freq is the frequency of the minor allele (a)
        r_freq = float(minor_allele_counts[r]) / float(total_alleles)
        # Is the recessive lethal? Yes?
        if recessives[r][2] == 1:
            # Compute the frequency of the AA and Aa genotypes
            denom = (1. - r_freq)**2 + (2 * r_freq * (1. - r_freq))
            f_dom = (1. - r_freq)**2 / denom
            f_het = (2 * r_freq * (1. - r_freq)) / denom
            print
            print '\tRecessive %s (%s), generation %s:' % (r, recessives[r][3], generation)
            print '\t\tminor alleles = %s\t\ttotal alleles = %s' % (minor_allele_counts[r], total_alleles)
            print '\t\tp = %s\t\tq = %s' % ((1. - r_freq), r_freq)
            print '\t\t  = %s\t\t  = %s' % ((1. - r_freq) - (1. - recessives[r-1][0]),
                                            r_freq - recessives[r-1][0])
            print '\t\tf(AA) = %s\t\tf(Aa) = %s' % (f_dom, f_het)
        # Well, okay, so it's not.
        else:
            # Compute the frequency of the AA and Aa genotypes
            f_dom = (1. - r_freq)**2
            f_het = (2 * r_freq * (1. - r_freq))
            f_rec = r_freq**2
            print
            print '\tThis recessive is ***NOT LETHAL***'
            print '\tRecessive %s (%s), generation %s:' % (r, recessives[r][3], generation)
            print '\t\tminor alleles = %s\t\ttotal alleles = %s' % (minor_allele_counts[r], total_alleles)
            print '\t\tp = %s\t\tq = %s' % ((1. - r_freq), r_freq)
            print '\t\t  = %s\t\t  = %s' % ((1. - r_freq) - (1. - recessives[r-1][0]),
                                            r_freq - recessives[r-1][0])
            print '\t\tf(AA) = %s\t\tf(Aa) = %s' % (f_dom, f_het)
            print '\t\tf(aa) = %s' % f_rec
        # Finally, update the recessives and history tables
        recessives[r][0] = r_freq
        freq_hist[generation].append(r_freq)
    return recessives, freq_hist


# We're going to go ahead and write files containing various pieces
# of information from the simulation.


def write_history_files(cows, bulls, dead_cows, dead_bulls, generation, filetag=''):
    # First, write the animal history files.
    cowfile = 'cows_history%s_%s.txt' % (filetag, generation)
    deadcowfile = 'dead_cows_history%s_%s.txt' % (filetag, generation)
    bullfile = 'bulls_history%s_%s.txt' % (filetag, generation)
    deadbullfile = 'dead_bulls_history%s_%s.txt' % (filetag, generation)
    # Column labels
    # next_id, bull_id, cow_id, generation, sex, 'A', '', -1, tbv, []
    headerline = 'animal\tsire\tdam\tborn\tsex\therd\tstatus\tcause\tdied\tTBV\trecessives\n'
    # Cows
    ofh = file(cowfile, 'w')
    ofh.write(headerline)
    for c in cows:
        outline = ''
        for p in c:
            outline += '\t%s' % p
        outline += '\n'
        ofh.write(outline)
    ofh.close()
    # Dead cows
    ofh = file(deadcowfile, 'w')
    ofh.write(headerline)
    for c in dead_cows:
        outline = ''
        for p in c:
            outline += '\t%s' % p
        outline += '\n'
        ofh.write(outline)
    ofh.close()
    # Bulls
    ofh = file(bullfile, 'w')
    ofh.write(headerline)
    for b in bulls:
        outline = ''
        for p in b:
            outline += '\t%s' % p
        outline += '\n'
        ofh.write(outline)
    ofh.close()
    # Dead bulls
    ofh = file(deadbullfile, 'w')
    ofh.write(headerline)
    for b in dead_bulls:
        outline = ''
        for p in b:
            outline += '\t%s' % p
        outline += '\n'
        ofh.write(outline)
    ofh.close()


# Main loop for individual simulation scenarios.
#
# scenario: the mating strategy to use in the current scenario
#           ( random | toppct | pryce )
# gens              : number of generations to run the simulation
# percent           : percent of bulls to use as sires in the toppct
#                     scenario
# base_bulls        : The number of bulls in the base population
# base_cows         : The number of cows in the base population
# base_herds        : Number of herds in the population.
# max_bulls         : The maximum number of bulls that can be alive at one time
# max_cows          : The maximum number of cows that can be alive at one time
# debug             : Flag to activate/deactivate debugging messages
# filetag           : String added to a filename to better describe what analysis
#                     a file is associated with
# recessives        : A Python list of recessives in the population
# max_matings   : The maximum number of matings permitted for each bull


def run_scenario(scenario='random', gens=20, percent=0.10, base_bulls=500, base_cows=2500,
                 base_herds=100, max_bulls=1500, max_cows=7500, debug=False, filetag='',
                 recessives=[], max_matings=500):

    # This is the initial setup
    cows, bulls, dead_cows, dead_bulls, freq_hist = setup(base_bulls=base_bulls,
                                                          base_cows=base_cows,
                                                          base_herds=base_herds,
                                                          recessives=recessives)

    # This is the start of the next generation
    for generation in xrange(1, gens+1):
        # First, mate the animals, which creates new offspring
        print
        print 'Generation %s' % generation
        print '\t              \tLC\tLB\tLT\tDC\tDB\tDT'
        print '\tBefore mating:\t%s\t%s\t%s\t%s\t%s\t%s' % (len(cows), len(bulls),
                                                            len(cows)+len(bulls),
                                                            len(dead_cows), len(dead_bulls),
                                                            len(dead_cows)+len(dead_bulls))

        # This is the code that handles the mating scenarios

        # Animals are mated at random with an [optional] limit on the number of matings
        # allowed to each bull.
        if scenario == 'random':
            cows, bulls, dead_cows, dead_bulls = random_mating(cows, bulls,
                                                               dead_cows, dead_bulls, generation,
                                                               recessives, max_matings=500)
        # Only the top "pct" of bulls, based on TBV, are mater randomly to the cow
        # population with no limit on the number of matings allowed. This is a simple
        # example of truncation selection.
        elif scenario == 'toppct':
            cows, bulls, dead_cows, dead_bulls = toppct_mating(cows, bulls,dead_cows,
                                                               dead_bulls, generation,
                                                               recessives, pct=percent)
        # Bulls are mated to cows using a mate allocation strategy similar to that of
        # Pryce et al. (2012), in which the PA is discounted to account for decreased
        # fitness associated with increased rates of inbreeding. We're not using genomic
        # information in this study but we assume perfect pedigrees, so everything should
        # work out okay.
        elif scenario == 'pryce':
            cows, bulls, dead_cows, dead_bulls = pryce_mating(cows, bulls,
                                                              dead_cows, dead_bulls,
                                                              generation, recessives,
                                                              max_matings=max_matings,
                                                              base_herds=base_herds,
                                                              debug=debug)
        # The default scenario is random mating.
        else:
            cows, bulls, dead_cows, dead_bulls = random_mating(cows, bulls,
                                                               dead_cows, dead_bulls,
                                                               generation, recessives,
                                                               max_matings=500)
    
        print '\tAfter mating:\t%s\t%s\t%s\t%s\t%s\t%s' % (len(cows), len(bulls),
                                                           len(cows)+len(bulls), len(dead_cows),
                                                           len(dead_bulls), len(dead_cows)+len(dead_bulls))
    
        # Cull bulls
        bbull_count, bbull_min, bbull_max, bbull_mean, bbull_var, bbull_std = animal_summary(bulls)
        bulls, dead_bulls = cull_bulls(bulls, dead_bulls, generation, max_bulls, debug=debug)
        abull_count, abull_min, abull_max, abull_mean, abull_var, abull_std = animal_summary(bulls)

        # Cull cows
        bcow_count, bcow_min, bcow_max, bcow_mean, bcow_var, bcow_std = animal_summary(cows)
        cows, dead_cows = cull_cows(cows, dead_cows, generation, max_cows, debug=debug)
        acow_count, acow_min, acow_max, acow_mean, acow_var, acow_std = animal_summary(cows)

        print '\tAfter culling:\t%s\t%s\t%s\t%s\t%s\t%s' % (len(cows), len(bulls), len(cows)+len(bulls),
                                                            len(dead_cows), len(dead_bulls),
                                                            len(dead_cows)+len(dead_bulls))

        print
        print '\tSummary statistics for TBV'
        print '\t--------------------------'
        print '\t    \t    \tN\tMin\t\tMax\t\tMean\t\tStd'
        print '\tBull\tpre \t%s\t%s\t%s\t%s\t%s' % (int(bbull_count), bbull_min, bbull_max, bbull_mean, bbull_std)
        print '\tBull\tpost\t%s\t%s\t%s\t%s\t%s' % (int(abull_count), abull_min, abull_max, abull_mean, abull_std)
        print '\tCow \tpre \t%s\t%s\t%s\t%s\t%s' % (int(bcow_count), bcow_min, bcow_max, bcow_mean, bcow_std)
        print '\tCow \tpost\t%s\t%s\t%s\t%s\t%s' % (int(acow_count), acow_min, acow_max, acow_mean, acow_std)

        # Now update the MAF for the recessives in the population
        recessives, freq_hist = update_maf(cows, bulls, generation, recessives, freq_hist)

        # Write history files
        write_history_files(cows, bulls, dead_cows, dead_bulls, generation, filetag)

    # Save the simulation parameters so that we know what we did.
    outfile = 'simulation_parameters%s.txt' % filetag
    ofh = file(outfile, 'w')
    outline = 'scenario    :\t%s\n' % scenario
    ofh.write(outline)
    outline = 'percent     :\t%s\n' % percent
    ofh.write(outline)
    outline = 'base bulls  :\t%s\n' % base_bulls
    ofh.write(outline)
    outline = 'base cows   :\t%s\n' % base_cows
    ofh.write(outline)
    outline = 'max bulls   :\t%s\n' % max_bulls
    ofh.write(outline)
    outline = 'max cows    :\t%s\n' % max_cows
    ofh.write(outline)
    for r in xrange(len(recessives)):
        outline = 'Base MAF %s :\t%s\n' % (r+1, recessives[r][0])
        ofh.write(outline)
        outline = 'Cost %s     :\t%s\n' % (r+1, recessives[r][1])
        ofh.write(outline)
    ofh.close()

    # Save the allele frequency history
    outfile = 'minor_allele_frequencies%s.txt' % filetag
    ofh = file(outfile, 'w')
    for k, v in freq_hist.iteritems():
        outline = '%s' % k
        for frequency in v:
            outline += '\t%s' % frequency
        outline += '\n'
        ofh.write(outline)
    ofh.close()

    # Now that we're done with the simulation let's go ahead and visualize the change in minor allele frequency
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("MAF change over time")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Minor allele frequency")
    x = freq_hist.keys()
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'y', 'm', 'k'])
    markers = itertools.cycle(['o', 's', 'v'])
    for r in xrange(len(recessives)):
        y = []
        for v in freq_hist.values():
            y.append(v[r])
        ax.plot(x, y, color=colors.next(), marker=markers.next(), label=recessives[r][3])
    ax.legend(loc='best')
    filename = "MAF_plot_%s_gen_%s_rec_%s.png" % (gens, len(recessives), scenario)
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':


    # Simulation parameters
    base_bulls = 150         # Initial number of founder bulls in the population
    base_cows = 3000         # Initial number of founder cows in the population
    base_herds = 15         # Number of herds in the population
    max_bulls = 600         # Maximum number of live bulls to keep each generation
    max_cows = 12000         # Maximum number of live cows to keep each generation
    percent = 0.10       # Proportion of bulls to use in the toppct scenario
    generations = 10          # How long to run the simulation
    max_matings = 200          # The maximum number of matings permitted for each bull
    debug = False          # Activate (True) or deactivate (False) debugging messages

    # Recessives are stored in a list of lists. The first value in each list
    # is the minor allele frequency in the base population, and the second
    # number is the economic value of the minor allele. If the economic value
    # is $20, that means that the value of an affected homozygote is -$20. The
    # third value in the record indicates if the recessive is lethal (0) or
    # non-lethal (0). The fourth value is a label that is not used for any
    # calculations.
    default_recessives = [
        [0.0276, 150, 1, 'Brachyspina'],
        [0.0192,  40, 1, 'HH1'],
        [0.0166,  40, 1, 'HH2'],
        [0.0295,  40, 1, 'HH3'],
        [0.0037,  40, 1, 'HH4'],
        [0.0222,  40, 1, 'HH5'],
        [0.0025, 150, 1, 'BLAD'],
        [0.0137,  70, 1, 'CVM'],
        [0.0001,  40, 1, 'DUMPS'],
        [0.0007, 150, 1, 'Mulefoot'],
        [0.71,   -20, 0, 'Polled'],
        [0.0542,  20, 0, 'Red'],
    ]

    # First, run the random mating scenario
#    print '=' * 80
#    recessives = copy.copy(default_recessives)
#    run_scenario(scenario='random', base_bulls=base_bulls, base_cows=base_cows, \
#        max_bulls=max_bulls, max_cows=max_cows, filetag='_ran_20_gen_1_rec_polled', \
#        recessives=recessives)

    # Now run truncation selection, just to introduce some genetic trend.
#    print '=' * 80
#    recessives = copy.copy(default_recessives)
#    run_scenario(scenario='toppct', percent=percent, base_bulls=base_bulls, \
#        base_cows=base_cows, max_bulls=max_bulls, max_cows=max_cows, \
#        filetag='_toppct_20_gen_1_rec_polled', recessives=recessives)

    # This is the real heart of the analysis, applying Pryce's method.
    print '=' * 80
    recessives = copy.copy(default_recessives)
    run_scenario(scenario='pryce', percent=percent, base_bulls=base_bulls, base_cows=base_cows,
                 base_herds=base_herds, max_bulls=max_bulls, max_cows=max_cows, debug=debug,
                 filetag='_pryce_20_gen_12_rec', recessives=recessives, gens=generations,
                 max_matings=max_matings)
