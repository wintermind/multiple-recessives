# The objective of this simulation is to consider several scenarios for the management of multiple
# recessive alleles in a simulated population of dairy cattle. The basic idea is simple:
# + Each animal has parents, a sex code, a true breeding value for lifetime net merit, and a genotype
#   for the recessive alleles in the population;
# + Each recessive has a minor allele frequency in the base population and an economic value;
# + Matings will be based on parent averages, and at-risk matings will be penalized by the economic
#   value of each recessive.


# Import standard libraries
import copy
import datetime
import itertools
import math
import random
import subprocess
import sys
import time

# Import external libraries
import matplotlib
# Force matplotlib to not use any X-windows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist
import numpy as np
import numpy.ma as ma
from scipy.stats import bernoulli
import ast

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
#    rng_kseed:             An integer used to seed the random number generation. If
#                           None, a random seed as described in the Python documentation
#                           is used.
#    debug                  Flag to activate/deactivate debugging messages.


def setup(base_bulls=500, base_cows=2500, base_herds=100, force_carriers=True, force_best=True,
          recessives=[], check_tbv=False, rng_seed=None, debug=True):

    # Base population parameters
    generation = 0                  # The simulation starts at generation 0. It's as though we're all C programmers.

    # Recessives are required since that's the whole point of this.
    if len(recessives) == 0:
        print '[setup]: The recessives dictionary passed to the setup() subroutine was empty! The program \
            cannot continue, and will halt.'
        sys.exit(1)

    # Seed the RNG
    if rng_seed:
        try:
            np.random.seed(int(rng_seed))
        except:
            np.random.seed()
    else:
        np.random.seed()

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

    # Add animals to the base cow list.
    if debug:
        print '[setup]: Adding animals to the base cow list at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for i in xrange(base_cows):
        # The list contents are:
        # animal ID, sire ID, dam ID, generation, sex, herd, alive/dead, reason dead, when dead, TBV,
        # # coefficient of inbreeding, and genotype
        # "generation" is the generation in which the base population animal was born, not its actual
        # age.
        c = i + 1
        if c in id_list:
            if debug:
                print '[setup]: Error! A cow with ID %s already exists in the ID list!' % c
        c_list = [c, 0, 0, (-1*random.randint(0, 4)), 'F', random.randint(0, base_herds), 'A',
                  '', -1, base_cow_tbv.item(i), 0.0, []]
        for r in xrange(len(recessives)):
            c_list[-1].append(base_cow_gt.item(i, r))
        cows.append(c_list)
        id_list.append(c)

    # Add animals to the bull list.
    if debug:
        print '[setup]: Adding animals to the base bull list at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for i in xrange(base_bulls):
        b = i + 1 + base_cows
        if b in id_list:
            if debug:
                print '[setup]: Error! A bull with ID %s already exists in the ID list!' % b
        b_list = [b, 0, 0, (-1*random.randint(0, 9)), 'M', random.randint(0, base_herds), 'A', '',
                  -1, base_bull_tbv.item(i), 0.0, []]
        for r in xrange(len(recessives)):
            b_list[-1].append(base_bull_gt.item(i, r))
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
    # Make sure we have current coefficients of inbreeding
    cows, bulls, dead_cows, dead_bulls = compute_inbreeding(cows, bulls, dead_cows, dead_bulls, generation, debug)
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
        print '[toppct_mating]: %s is outside of the range 0.0 <= pct <= 1.0, changing to 0.10' % pct
        pct = 0.10
    # Make sure we have current coefficients of inbreeding
    cows, bulls, dead_cows, dead_bulls = compute_inbreeding(cows, bulls, dead_cows, dead_bulls, generation, debug)
    # Sort bulls on TBV in ascending order
    bulls.sort(key=lambda x: x[9])
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


# compute_inbreeding() INBUPGF90 to compute coefficients of inbreeding for each animal
# in the pedigree, and updated animal records to include that information.
#
# cows          : A list of live cow records
# bulls         : A list of live bull records
# dead_cows     : A list of dead cow records
# dead_bulls    : A list of dead bull records
# debug         : Flag to activate/deactivate debugging messages
# generation    : The current generation in the simulation


def compute_inbreeding(cows, bulls, dead_cows, dead_bulls, generation, debug=False):

    #
    # Now, we're going to need to construct a pedigree that includes matings of all cows in
    # each herd to the bulls randomly assigned to that herd. Bulls are randomly assigned to
    # herds to reflect different sire selection policies. It is faster to calculate the
    # inbreeding of the potential offspring than it is to calculate relationships among parents
    # because the latter requires that we store relationships among all parents.
    #
    # I'm going to try and allocate a fixed-length NumPy array large enough to store the entire
    # pedigree to avoid memory fragmentation/swapping when working with large Python lists. If that
    # can't be done, I'll fall back to a list. This does mean that the code has to accommodate both
    # cases, so it's a little verbose.
    next_id = get_next_id(cows, bulls, dead_cows, dead_bulls)
    if debug:
        print '\t[compute_inbreeding]: next_id = %s in generation %s' % (next_id, generation)
    pedigree_size = len(cows) + len(dead_cows) + len(bulls) + len(dead_bulls)
    try:
        pedigree = np.zeros((pedigree_size,), dtype=('a20, a20, a20, i4'))
        print '\t[compute_inbreeding]: Allocated a NumPy array of size %s to store pedigree at %s' % \
              (pedigree_size, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    except MemoryError:
        pedigree = []
        print '\t[compute_inbreeding]: Could not allocate an array of size %s, using a SLOWWWW Python list at %s' % \
              (pedigree_size, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    id_list = []
    pedigree_counter = 0
    pedigree_array = isinstance(pedigree, (np.ndarray, np.generic))
    if debug:
        print '\t[compute_inbreeding]: Putting all cows and bulls in a pedigree at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for c in cows:
        if pedigree_array:
            pedigree[pedigree_counter] = (c[0], c[1], c[2], c[3]+10)
        else:
            pedigree.append(' '.join([c[0], c[1], c[2], c[3]+10, '\n']))
            if c[0] not in id_list:
                id_list.append(c[0])
        pedigree_counter += 1
    for dc in dead_cows:
        if pedigree_array:
            pedigree[pedigree_counter] = (dc[0], dc[1], dc[2], dc[3]+10)
        else:
            pedigree.append(' '.join([dc[0], dc[1], dc[2], dc[3]+10, '\n']))
            if dc[0] not in id_list:
                id_list.append(dc[0])
        pedigree_counter += 1
    for b in bulls:
        if pedigree_array:
            pedigree[pedigree_counter] = (b[0], b[1], b[2], b[3]+10)
        else:
            pedigree.append(' '.join([b[0], b[1], b[2], b[3]+10, '\n']))
            if b[0] not in id_list:
                id_list.append(b[0])
        pedigree_counter += 1
    for db in dead_bulls:
        if isinstance(pedigree, (np.ndarray, np.generic)):
            pedigree[pedigree_counter] = (db[0], db[1], db[2], db[3]+10)
        else:
            pedigree.append(' '.join([db[0], db[1], db[2], db[3]+10, '\n']))
            if db[0] not in id_list:
                id_list.append(db[0])
        pedigree_counter += 1
    if debug:
        print '\t[compute_inbreeding]: %s "old" animals in pedigree in generation %s at %s' % \
            (pedigree_counter, generation, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Write the pedigree to a file.
    pedfile = 'compute_inbreeding_%s.txt' % generation
    if debug:
        print '\t[compute_inbreeding]: Writing pedigree to %s at %s' % \
              (pedfile, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    ofh = file(pedfile, 'w')
    if isinstance(pedigree, (np.ndarray, np.generic)):
        for pidx in xrange(pedigree_counter):
            p = ' '.join([pedigree[pidx][0], pedigree[pidx][1], pedigree[pidx][2], str(pedigree[pidx][3]), '\n'])
            ofh.write(p)
    else:
        for p in pedigree:
            ofh.write(p)
    ofh.close()
    del pedigree

    # PyPedal is just too slow when the pedigrees are large (e.g., millons of records), so
    # we're going to use Ignacio Aguilar's INBUPGF90 program.
    #
    # Per an e-mail from Ignacio Aguilar on 06/25/2014, INBUPGF90 does NOT emit a proper
    # status return code when it exits, which makes it tricky to know for sure when the
    # job is done. I've observed a number of cases where the simulation appears to stall
    # because subprocess.call() does not recognize that INBUPGF90 has finished a job. So,
    # I've cobbled-together a solution using ideas from Ezequiel Nicolazzi
    # (https://github.com/nicolazzie/AffyPipe/blob/master/AffyPipe.py) and a post on
    # Stack Overflow (http://stackoverflow.com/questions/12057794/
    # python-using-popen-poll-on-background-process). I'm not 100% sure that this works
    # as intended, but I'm out of ideas.
    logfile = 'compute_inbreeding_%s.log' % generation
    # Several methods can be used:
    # 1 - recursive as in Aguilar & Misztal, 2008 (default)
    # 2 - recursive but with coefficients store in memory, faster with large number of
    #     generations but more memory requirements
    # 3 - method as in Meuwissen & Luo 1992
    if debug:
        print '\t[compute_inbreeding]: Started inbupgf90 to calculate COI at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    callinbupgf90 = ['inbupgf90', '--pedfile', pedfile, '--method', '3', '--yob', '>', logfile, '2>&1&']
    time_waited = 0
    p = subprocess.Popen(callinbupgf90, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while p.poll() is None:
        # Wait 1 second between pokes with a sharp stick.
        time.sleep(10)
        time_waited += 10
        p.poll()
        if time_waited % 60 == 0 and debug:
            print '\t\t[compute_inbreeding]: Waiting for INBUPGF90 to finish -- %s minutes so far...' % int(time_waited/60)
    # Pick-up the output from INBUPGF90
    (results, errors) = p.communicate()
    if debug:
        if errors == '':
            print '\t\t[compute_inbreeding]: INBUPGF90 finished without problems at %s!' % \
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            if debug:
                print '\t\t\t%s' % results
        else:
            print '\t\t[compute_inbreeding]: errors: %s' % errors
        print '\t[compute_inbreeding]: Finished inbupgf90 to calculate COI at %s' %\
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load the COI into a dictionary keyed by original animal ID
    coifile = 'compute_inbreeding_%s.txt.solinb' % generation
    if debug:
        print '\t[compute_inbreeding]: Putting coefficients of inbreeding from %s.solinb in a dictionary at %s' \
            % (pedfile, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    inbr = {}
    ifh = open(coifile, 'r')
    for line in ifh:
        pieces = line.split()
        inbr[int(pieces[0])] = float(pieces[1])
    ifh.close()

    # Now, assign the coefficients of inbreeding to the animal records
    for c in cows: c[10] = inbr[c[0]]
    for dc in dead_cows: dc[10] = inbr[dc[0]]
    for b in bulls: b[10] = inbr[b[0]]
    for db in dead_bulls: db[10] = inbr[db[0]]

    # Send everything back to the calling routine
    return cows, bulls, dead_cows, dead_bulls

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
# penalty       : If True, adjust PA for recessives, else adjust only for inbreeding


def pryce_mating(cows, bulls, dead_cows, dead_bulls, generation,
                 recessives, max_matings=500, base_herds=100, debug=False,
                 penalty=False):

    if debug:
        print '\t[pryce_mating]: Parameters:\n\t\tgeneration: %s\n\t\tmax_matings: %s\n\t\tbase_herds: ' \
              '%s\n\t\tdebug: %s\n\t\tRecessives:' % (generation, max_matings, base_herds, debug)
        for r in recessives:
            print '\t\t\t%s' % r
    # Never trust users, they are lying liars
    if max_matings < 0:
        print '\t[pryce_mating]: %s is less than 0, changing num_matings to 500.' % max_matings
        max_matings = 500
    if not type(max_matings) is int:
        print '\t[pryce_mating]: % is not not an integer, changing num_matings to 500.' % max_matings
    #
    # Now, we're going to need to construct a pedigree that includes matings of all cows in
    # each herd to the bulls randomly assigned to that herd. Bulls are randomly assigned to
    # herds to reflect different sire selection policies. It is faster to calculate the
    # inbreeding of the potential offspring than it is to calculate relationships among parents
    # because the latter requires that we store relationships among all parents.
    #
    # I'm going to try and allocate a fixed-length NumPy array large enough to store the entire
    # pedigree to avoid memory fragmentation/swapping when working with large Python lists. If that
    # can't be done, I'll fall back to a list. This does mean that the code has to accommodate both
    # cases, so it's a little verbose.
    next_id = get_next_id(cows, bulls, dead_cows, dead_bulls)
    if debug:
        print '\t[pryce_mating]: next_id = %s in generation %s' % (next_id, generation)
    matings = {}
    bull_portfolio = {}
    cow_portfolio = {}
    pedigree_size = len(cows) + len(dead_cows) + len(bulls) + len(dead_bulls)
    # Note that I'm including a fudge factor by multiplying the bulls by 2 so that we get an
    # array longer than we need.
    pedigree_size += int((2*len(bulls)) * len(cows)) / base_herds
    try:
        pedigree = np.zeros((pedigree_size,), dtype=('a20, a20, a20, i4'))
        print '\t[pryce_mating]: Allocated a NumPy array of size %s to store pedigree at %s' % \
              (pedigree_size, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    except MemoryError:
        pedigree = []
        print '\t[pryce_mating]: Could not allocate an array of size %s, using a SLOWWWW Python list at %s' % \
              (pedigree_size, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    id_list = []
    pedigree_counter = 0
    pedigree_array = isinstance(pedigree, (np.ndarray, np.generic))
    if debug:
        print '\t[pryce_mating]: Putting all cows and herd bulls in a pedigree at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for c in cows:
        if pedigree_array:
            pedigree[pedigree_counter] = (c[0], c[1], c[2], c[3]+10)
        else:
            pedigree.append(' '.join([c[0], c[1], c[2], c[3]+10, '\n']))
            if c[0] not in id_list:
                id_list.append(c[0])
        pedigree_counter += 1
    for dc in dead_cows:
        if pedigree_array:
            pedigree[pedigree_counter] = (dc[0], dc[1], dc[2], dc[3]+10)
        else:
            pedigree.append(' '.join([dc[0], dc[1], dc[2], dc[3]+10, '\n']))
            if dc[0] not in id_list:
                id_list.append(dc[0])
        pedigree_counter += 1
    for b in bulls:
        if pedigree_array:
            pedigree[pedigree_counter] = (b[0], b[1], b[2], b[3]+10)
        else:
            pedigree.append(' '.join([b[0], b[1], b[2], b[3]+10, '\n']))
            if b[0] not in id_list:
                id_list.append(b[0])
        pedigree_counter += 1
        matings[b[0]] = 0
    for db in dead_bulls:
        if isinstance(pedigree, (np.ndarray, np.generic)):
            pedigree[pedigree_counter] = (db[0], db[1], db[2], db[3]+10)
        else:
            pedigree.append(' '.join([db[0], db[1], db[2], db[3]+10, '\n']))
            if db[0] not in id_list:
                id_list.append(db[0])
        pedigree_counter += 1
    if pedigree_array:
        id_list = pedigree[:][0].tolist()
    if debug:
        print '\t[pryce_mating]: %s "old" animals in pedigree in generation %s at %s' % \
            (pedigree_counter, generation, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # Now we need to create faux offspring of the living bulls and cows because it is faster to
    # compute inbreeding than relationships.
    calfcount = 0
    n_bulls = int(round(len(bulls) / base_herds, 0))
    if debug:
        #print '\t[pryce_mating]: base_herds: %s' % base_herds
        #print '\t[pryce_mating]: len(bulls): %s' % len(bulls)
        #print '\t[pryce_mating]: len(bulls) / base_herds: %s' % (float(len(bulls)) / float(base_herds))
        #print '\t[pryce_mating]: round(len(bulls) / base_herds, 0): %s' % (round(len(bulls) / base_herds, 0))
        #print '\t[pryce_mating]: int(round(len(bulls) / base_herds, 0)): %s' % (int(round(len(bulls) / base_herds, 0)))
        #print '\t[pryce_mating]: n_bulls: %s' % n_bulls
        print '\t[pryce_mating]: Mating all cows to all herd bulls at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for herd in xrange(base_herds):
        bull_portfolio[herd] = []
        cow_portfolio[herd] = []
        #if debug:
        #    print '\t\t[pryce_mating]: Shuffling bulls at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        random.shuffle(bulls)               # Randomly assign bulls to cows
        herd_bulls = bulls[0:n_bulls+1]
        herd_cows = [c for c in cows if c[5] == herd]
        #if debug:
        #    print '\t\t[pryce_mating]: Beginning loop to make calves for herd %s at %s' %\
        #          (herd, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        for b in herd_bulls:
            bull_portfolio[herd].append(b)
            for c in herd_cows:
                cow_portfolio[herd].append(c)
                calf_id = str(b[0])+'__'+str(c[0])
                if calf_id in id_list:
                    if debug:
                        print '\t\t[pryce_mating]: Error! A calf with ID %s already exists in the ID list in \
                            generation %s!' % (calf_id, generation)
                if pedigree_array:
                    pedigree[pedigree_counter] = (calf_id, b[0], c[0], generation+10)
                else:
                    pedigree.append(' '.join([calf_id, b[0], c[0], generation+10, '\n']))
                pedigree_counter += 1
                calfcount += 1
        #if debug:
        #    print '\t\t[pryce_mating]: Finished loop to make calves for herd %s at %s' % \
        #          (herd, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    if debug:
        print '\t\t[pryce_mating]: %s calves added to pedigree in generation %s at %s' % \
            (calfcount, generation, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print '\t\t[pryce_mating]: %s total animals in pedigree in generation %s at %s' % \
            (pedigree_counter, generation, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    # Write the pedigree to a file.
    if penalty:
        pedfile = 'pedigree_pryce_%s.txt' % generation
    else:
        pedfile = 'pedigree_pryce_r_%s.txt' % generation
    if debug:
        print '\t[pryce_mating]: Writing pedigree to %s at %s' % \
              (pedfile, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    ofh = file(pedfile, 'w')
    if isinstance(pedigree, (np.ndarray, np.generic)):
        for pidx in xrange(pedigree_counter):
            p = ' '.join([pedigree[pidx][0], pedigree[pidx][1], pedigree[pidx][2], str(pedigree[pidx][3]), '\n'])
            ofh.write(p)
    else:
        for p in pedigree:
            ofh.write(p)
    ofh.close()
    del pedigree
    # PyPedal is just too slow when the pedigrees are large (e.g., millons of records), so
    # we're going to use Ignacio Aguilar's INBUPGF90 program.
    #
    # Per an e-mail from Ignacio Aguilar on 06/25/2014, INBUPGF90 does NOT emit a proper
    # status return code when it exits, which makes it tricky to know for sure when the
    # job is done. I've observed a number of cases where the simulation appears to stall
    # because subprocess.call() does not recognize that INBUPGF90 has finished a job. So,
    # I've cobbled-together a solution using ideas from Ezequiel Nicolazzi
    # (https://github.com/nicolazzie/AffyPipe/blob/master/AffyPipe.py) and a post on
    # Stack Overflow (http://stackoverflow.com/questions/12057794/
    # python-using-popen-poll-on-background-process). I'm not 100% sure that this works
    # as intended, but I'm out of ideas.
    if penalty:
        logfile = 'pedigree_pryce_%s.log' % generation
    else:
        logfile = 'pedigree_pryce_r_%s.log' % generation
    # Several methods can be used:
    # 1 - recursive as in Aguilar & Misztal, 2008 (default)
    # 2 - recursive but with coefficients store in memory, faster with large number of
    #     generations but more memory requirements
    # 3 - method as in Meuwissen & Luo 1992
    if debug:
        print '\t[pryce_mating]: Started inbupgf90 to calculate COI at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    callinbupgf90 = ['inbupgf90', '--pedfile', pedfile, '--method', '3', '--yob', '>', logfile, '2>&1&']
    time_waited = 0
    p = subprocess.Popen(callinbupgf90, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while p.poll() is None:
        # Wait 1 second between pokes with a sharp stick.
        time.sleep(10)
        time_waited += 10
        p.poll()
        if time_waited % 60 == 0 and debug:
            print '\t\t[pryce_mating]: Waiting for INBUPGF90 to finish -- %s minutes so far...' % int(time_waited/60)
    # Pick-up the output from INBUPGF90
    (results, errors) = p.communicate()
    if debug:
        if errors == '':
            print '\t\t[pryce_mating]: INBUPGF90 finished without problems at %s!' % \
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            #if debug:
            #    print '\t\t\t%s' % results
        else:
            print '\t\t[pryce_mating]: errors: %s' % errors
        print '\t[pryce_mating]: Finished inbupgf90 to calculate COI at %s' %\
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Load the COI into a dictionary keyed by original animal ID
    if penalty:
        coifile = 'pedigree_pryce_%s.txt.solinb' % generation
    else:
        coifile = 'pedigree_pryce_r_%s.txt.solinb' % generation
    if debug:
        print '\t[pryce_mating]: Putting coefficients of inbreeding from %s.solinb in a dictionary at %s' \
            % (pedfile, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    inbr = {}
    ifh = open(coifile, 'r')
    for line in ifh:
        pieces = line.split()
        inbr[pieces[0]] = float(pieces[1])
    ifh.close()

    # Now, assign the coefficients of inbreeding to the "old" animal records
    if debug:
        print '\t[pryce_mating]: Writing coefficients of inbreeding to animal records at %s' \
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    for c in cows: c[10] = inbr[str(c[0])]
    for dc in dead_cows: dc[10] = inbr[str(dc[0])]
    for b in bulls: b[10] = inbr[str(b[0])]
    for db in dead_bulls: db[10] = inbr[str(db[0])]

    # We want to save F_ij and \sum{P(aa)} for individual matings for later analysis.
    if penalty:
        fpdict = {}

    flambda = 25.           # Loss of NM$ per 1% increase in inbreeding
    if debug:
        print '\t[pryce_mating]: Starting loop over herds to identify optimal matings at %s' % \
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # For each herd we're going to loop over all possible matings of the cows in the herd to the randomly chosen
    # bull portfolio and compute a parent average. Then we'll select the actual matings. This will be on a within-
    # herd basis, so a new B and M will be computed for each herd.
    for h in bull_portfolio.keys():
        # We need these lists so that we can step into the correct locations in the relationship matrix to get the
        # relationship of each cow to each bull.
        bids = [str(b[0]) for b in bull_portfolio[h]]
        cids = [str(c[0]) for c in cow_portfolio[h]]
        # Setup the B_0 matrix, which will contain PA BV plus an inbreeding penalty
        b_mat = ma.zeros((len(bull_portfolio[h]), len(cow_portfolio[h])))
        # Setup the F matrix, which will contain inbreeding coefficients
        f_mat = ma.zeros((len(bull_portfolio[h]), len(cow_portfolio[h])))
        # Setup the M matrix, which will contain the actual matings
        m_mat = ma.zeros((len(bull_portfolio[h]), len(cow_portfolio[h])))
        # Now process the herd the first time, to compute PA.
        for b in bull_portfolio[h]:
            bidx = bids.index(str(b[0]))
            for c in cow_portfolio[h]:
                cidx = cids.index(str(c[0]))
                calf_id = str(b[0])+'__'+str(c[0])
                # Set accumulator of \sum P(aa) to 0.
                if penalty:
                    paa_sum = 0.
                # If there is a hit on this key the bull was used in the herd in which the cow lives.
                try:
                    # Update the matrix of inbreeding coefficients.
                    f_mat[bidx, cidx] = inbr[calf_id]
                    # Now adjust the PA to account for inbreeding and the economic impacts of the recessives.
                    b_mat[bidx, cidx] = (0.5 * (b[9] + c[9])) - (inbr[calf_id] * 100 * flambda)
                    # Adjust the PA of the mating to account for recessives on. If the flag is not set then
                    # results should be similar to those of Pryce et al. (2012).
                    if penalty:
                        for r in xrange(len(recessives)):
                            # What are the parent genotypes?
                            b_gt = b[-1][r]
                            c_gt = c[-1][r]
                            if b_gt == -1 and c_gt == -1:           # aa genotypes
                                # Affected calf, adjust the PA by the full value of an aa calf.
                                b_mat[bidx, cidx] -= recessives[r][1]
                                paa_sum += 1.
                            elif b_gt == 1 and c_gt == 1:           # AA genotypes
                                # Calf cannot be aa, no adjustment to the PA.
                                pass
                            else:
                                # There is a 1/4 chance of having an affected calf,
                                # so the PA is adjusted by 1/4 of the "value" of an
                                # aa calf.
                                b_mat[bidx, cidx] -= (0.25 * recessives[r][1])
                                paa_sum += 0.25
                        # Store the inbreeding/P(aa) info for later. We're saving only calves because they're the animals
                        # for which we sum the P(aa) to make mating decisions.
                        fpdict[calf_id] = {}
                        fpdict[calf_id]['sire'] = str(b[0])
                        fpdict[calf_id]['dam'] = str(c[0])
                        fpdict[calf_id]['gen'] = generation
                        fpdict[calf_id]['inbr'] = inbr[calf_id]
                        fpdict[calf_id]['paa'] = paa_sum
                        fpdict[calf_id]['mating'] = 0
                # If there is not a matching key in the dictionary, then that cow-bull combination was not evaluated,
                # which means the bull was not used in the same herd as the cow.
                except KeyError:
                    b_mat[bidx, cidx] = ma.masked
                    f_mat[bidx, cidx] = ma.masked
        #
        # From Pryce et al. (2012) (http://www.journalofdairyscience.org/article/S0022-0302(11)00709-0/fulltext#sec0030)
        # A matrix of selected mates (mate allocation matrix; M) was constructed, where Mij=1 if the corresponding
        # element, Bij was the highest value in the column Bj; that is, the maximum value of all feasible matings for
        # dam j, all other elements were set to 0, and were rejected sire and dam combinations.
        #
        # Sort bulls on ID in ascending order
        bull_portfolio[h].sort(key=lambda x: x[0])
        cow_id_list = [c[0] for c in cow_portfolio[h]]
        new_bulls = []
        new_cows = []
        # Now loop over B to allocate the best matings
        for c in cow_portfolio[h]:
            if c[6] == 'A':
                # What column in b_mat corresponds to cow c?
                cow_loc = cow_id_list.index(c[0])
                # Get a vector of indices that would result in a sorted list.
                sorted_bulls = ma.argsort(b_mat[:, cow_loc])
                # The first element in sorted_bulls is the index of the smallest element in b_mat[:,cow_loc]. The
                # last element in sorted_bulls is the index of the largest element in b_mat[:,cow_loc].
                for bidx in xrange(len(bull_portfolio[h])-1, -1, -1):
                    # Does this bull still have matings available?
                    if matings[bull_portfolio[h][sorted_bulls[bidx]][0]] >= max_matings:
                        pass
                    elif bull_portfolio[h][sorted_bulls[bidx]][6] != 'A':
                        pass
                    else:
                        m_mat[sorted_bulls[bidx], cow_loc] = 1
                        matings[bull_portfolio[h][sorted_bulls[bidx]][0]] += 1
                        calf = create_new_calf(bull_portfolio[h][sorted_bulls[bidx]], c, recessives, next_id,
                                               generation, debug=debug)
                        calf_id = str(bull_portfolio[h][sorted_bulls[bidx]][0])+'__'+str(c[0])
                        fpdict[calf_id]['mating'] = 1
                        if calf[4] == 'F': new_cows.append(calf)
                        else: new_bulls.append(calf)
                        next_id += 1
                        # ...and, we're done.
                        break

    # Write the F_ij / \sum{P(aa)} information that we've been accumulating to a file for later analysis.
    if penalty:
        fpfile = 'fij_paa_pryce_%s.txt' % generation
        fph = open(fpfile, 'w')
        for fpkey in fpdict.keys():
            fpline = '%s %s %s %s %s %s %s\n' % ( fpkey, fpdict[calf_id]['sire'], fpdict[calf_id]['dam'],
                                                  fpdict[calf_id]['gen'], fpdict[calf_id]['inbr'],
                                                  fpdict[calf_id]['paa'], fpdict[calf_id]['mating']
                                                  )
            fph.write(fpline)
        fph.close()

    if debug:
        print '\t[pryce_mating]: Finished assigning mates and updating M_0 at %s' % \
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        print '\t\t[pryce_mating]: %s animals in original cow list' % len(cows)
        print '\t\t[pryce_mating]: %s animals in new cow list' % len(new_cows)
        print '\t\t[pryce_mating]: %s animals in original bull list' % len(bulls)
        print '\t\t[pryce_mating]: %s animals in new bull list' % len(new_bulls)
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
        print '\t\t[pryce_mating]: %s animals in final live cow list' % len(cows)
        print '\t\t[pryce_mating]: %s animals in final dead cow list' % len(dead_cows)
        print '\t\t[pryce_mating]: %s animals in final live bull list' % len(bulls)
        print '\t\t[pryce_mating]: %s animals in final dead bull list' % len(dead_bulls)

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
    #tbv += (random.normalvariate(0, 1) * 200)
    var_adj = math.sqrt(0.5) * ( 1. - ( 0.5 * ( sire[10] + dam[10] ) ) )
    tbv += (random.normalvariate(0, 1) * 200 * var_adj)
    # Form the animal record. Note that heifers are born into the same herd as their
    # dam.
    calf = [calf_id, sire[0], dam[0], generation, sex, dam[5], 'A', '', -1, tbv, 0.0, []]
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
        print '[cull_bulls]: dead bulls: %s' % len(dead_bulls)
    if max_bulls <= 0:
        print "[cull_bulls]: max_bulls cannot be <= 0! Setting to 250."
        max_bulls = 250
    if debug:
        print "[cull_bulls]: Computing age distribution."
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
    # Check to see if we need to cull on number (count).
    if len(bulls) <= max_bulls:
        if debug:
            print '\t[cull_bulls]: No bulls culled in generation %s (bulls<max_bulls)' % generation
        return bulls, dead_bulls
    # If this culling is necessary then we need to update the records of the
    # culled bulls and move them into the dead_bulls list. We cull bulls on
    # TBV.
    else:
        # Now we're going to sort on TBV in ascending order
        bulls.sort(key=lambda x: x[9])
        n_culled = 0
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
        if age not in ages.keys():
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
    if debug:
        print "[cull_cows]: Computing age distribution."
        age_distn(cows, generation)
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
#
# animals          : A list of animal records


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
#
# cows              : A list of live cow records
# bulls             : A list of live bull records
# generation        : The current generation in the simulation
# recessives        : A Python list of recessives in the population
# freq_hist         : A dictionary of minor allele frequencies for each generation
# show_recessives   : When True, print summary information for each recessive.


def update_maf(cows, bulls, generation, recessives, freq_hist, show_recessives=False):
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
            if show_recessives:
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
            if show_recessives:
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
#
# cows          : A list of live cow records
# bulls         : A list of live bull records
# dead_cows     : A list of dead cow records
# dead_bulls    : A list of dead bull records
# generation    : The current generation in the simulation
# filetag       : String added to a filename to better describe what analysis
#                 a file is associated with


def write_history_files(cows, bulls, dead_cows, dead_bulls, generation, filetag=''):
    # First, write the animal history files.
    cowfile = 'cows_history%s_%s.txt' % (filetag, generation)
    deadcowfile = 'dead_cows_history%s_%s.txt' % (filetag, generation)
    bullfile = 'bulls_history%s_%s.txt' % (filetag, generation)
    deadbullfile = 'dead_bulls_history%s_%s.txt' % (filetag, generation)
    # Column labels
    # next_id, bull_id, cow_id, generation, sex, 'A', '', -1, tbv, 0.0, []
    headerline = 'animal\tsire\tdam\tborn\tsex\therd\tstatus\tcause\tdied\tTBV\tinbreeding\trecessives\n'
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
# max_matings       : The maximum number of matings permitted for each bull
# show_recessives   : When True, print summary information for each recessive.


def run_scenario(scenario='random', gens=20, percent=0.10, base_bulls=500, base_cows=2500,
                 base_herds=100, max_bulls=1500, max_cows=7500, debug=False, filetag='',
                 recessives=[], max_matings=500, rng_seed=None, show_recessives=False):

    # This is the initial setup
    print '[run_scenario]: Setting-up the simulation at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cows, bulls, dead_cows, dead_bulls, freq_hist = setup(base_bulls=base_bulls,
                                                          base_cows=base_cows,
                                                          base_herds=base_herds,
                                                          recessives=recessives,
                                                          rng_seed=rng_seed)

    # This is the start of the next generation
    for generation in xrange(1, gens+1):
        print '\n[run_scenario]: Beginning generation %s at %s' % (
            generation, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        # First, mate the animals, which creates new offspring
        print
        print '\tGeneration %s' % generation
        print '\t\t              \tLC\tLB\tLT\tDC\tDB\tDT'
        print '\t\tBefore mating:\t%s\t%s\t%s\t%s\t%s\t%s' % \
              (len(cows), len(bulls), len(cows)+len(bulls),
               len(dead_cows), len(dead_bulls),
               len(dead_cows)+len(dead_bulls))

        # This is the code that handles the mating scenarios

        # Animals are mated at random with an [optional] limit on the number of matings
        # allowed to each bull.
        if scenario == 'random':
            print '\n[run_scenario]: Mating cows randomly at %s' % \
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            cows, bulls, dead_cows, dead_bulls = random_mating(cows, bulls,
                                                               dead_cows, dead_bulls, generation,
                                                               recessives, max_matings=500)
        # Only the top "pct" of bulls, based on TBV, are mater randomly to the cow
        # population with no limit on the number of matings allowed. This is a simple
        # example of truncation selection.
        elif scenario == 'toppct':
            print '\n[run_scenario]: Mating cows using truncation selection at %s' % \
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            cows, bulls, dead_cows, dead_bulls = toppct_mating(cows, bulls, dead_cows,
                                                               dead_bulls, generation,
                                                               recessives, pct=percent)
        # Bulls are mated to cows using a mate allocation strategy similar to that of
        # Pryce et al. (2012), in which the PA is discounted to account for decreased
        # fitness associated with increased rates of inbreeding. We're not using genomic
        # information in this study but we assume perfect pedigrees, so everything should
        # work out okay.
        elif scenario == 'pryce':
            print '\n[run_scenario]: Mating cows using Pryce\'s method at %s' % \
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            cows, bulls, dead_cows, dead_bulls = pryce_mating(cows, bulls,
                                                              dead_cows, dead_bulls,
                                                              generation, recessives,
                                                              max_matings=max_matings,
                                                              base_herds=base_herds,
                                                              debug=debug,
                                                              penalty=False)
        # Bulls are mated to cows using a mate allocation strategy similar to that of
        # Pryce et al. (2012), in which the PA is discounted to account for decreased
        # fitness associated with increased rates of inbreeding. We're not using genomic
        # information in this study but we assume perfect pedigrees, so everything should
        # work out okay. In addition, the PA are adjusted to account for the effects of
        # the recessives carried by the parents.
        elif scenario == 'pryce_r':
            print '\n[run_scenario]: Mating cows using Pryce\'s method and accounting for recessives at %s' % \
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            cows, bulls, dead_cows, dead_bulls = pryce_mating(cows, bulls,
                                                              dead_cows, dead_bulls,
                                                              generation, recessives,
                                                              max_matings=max_matings,
                                                              base_herds=base_herds,
                                                              debug=debug,
                                                              penalty=True)

        # The default scenario is random mating.
        else:
            cows, bulls, dead_cows, dead_bulls = random_mating(cows, bulls,
                                                               dead_cows, dead_bulls,
                                                               generation, recessives,
                                                               max_matings=500)

        print
        print '\t\t             \tLC\tLB\tLT\tDC\tDB\tDT'
        print '\t\tAfter mating:\t%s\t%s\t%s\t%s\t%s\t%s' % (len(cows), len(bulls),
                                                             len(cows)+len(bulls), len(dead_cows),
                                                             len(dead_bulls), len(dead_cows)+len(dead_bulls))
    
        # Cull bulls
        print '\n[run_scenario]: Computing summary statistics for bulls before culling at %s' %\
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        bbull_count, bbull_min, bbull_max, bbull_mean, bbull_var, bbull_std = animal_summary(bulls)
        print '\n[run_scenario]: Culling bulls at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        bulls, dead_bulls = cull_bulls(bulls, dead_bulls, generation, max_bulls, debug=debug)
        print '\n[run_scenario]: Computing summary statistics for bulls after culling at %s' %\
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        abull_count, abull_min, abull_max, abull_mean, abull_var, abull_std = animal_summary(bulls)

        # Cull cows
        print '\n[run_scenario]: Computing summary statistics for cows before culling at %s' % \
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        bcow_count, bcow_min, bcow_max, bcow_mean, bcow_var, bcow_std = animal_summary(cows)
        print '\n[run_scenario]: Culling cows at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        cows, dead_cows = cull_cows(cows, dead_cows, generation, max_cows, debug=debug)
        print '\n[run_scenario]: Computing summary statistics for cows after culling at %s' % \
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        acow_count, acow_min, acow_max, acow_mean, acow_var, acow_std = animal_summary(cows)

        print
        print '\t\t              \tLC\tLB\tLT\tDC\tDB\tDT'
        print '\t\tAfter culling:\t%s\t%s\t%s\t%s\t%s\t%s' % (len(cows), len(bulls), len(cows)+len(bulls),
                                                              len(dead_cows), len(dead_bulls),
                                                              len(dead_cows)+len(dead_bulls))

        print
        print '\t\tSummary statistics for TBV'
        print '\t\t--------------------------'
        print '\t\t    \t    \tN\tMin\t\tMax\t\tMean\t\tStd'
        print '\t\tBull\tpre \t%s\t%s\t%s\t%s\t%s' % (int(bbull_count), bbull_min, bbull_max, bbull_mean, bbull_std)
        print '\t\tBull\tpost\t%s\t%s\t%s\t%s\t%s' % (int(abull_count), abull_min, abull_max, abull_mean, abull_std)
        print '\t\tCow \tpre \t%s\t%s\t%s\t%s\t%s' % (int(bcow_count), bcow_min, bcow_max, bcow_mean, bcow_std)
        print '\t\tCow \tpost\t%s\t%s\t%s\t%s\t%s' % (int(acow_count), acow_min, acow_max, acow_mean, acow_std)

        # Now update the MAF for the recessives in the population
        print '\n[run_scenario]: Updating minor allele frequencies at %s' %\
              datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        recessives, freq_hist = update_maf(cows, bulls, generation, recessives, freq_hist,
                                           show_recessives)

        # Write history files
        print '\n[run_scenario]: Writing history files at %s' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
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
    outline = 'base herds  :\t%s\n' % base_herds
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
    outline = 'Debug       :\t%s\n' % debug
    ofh.write(outline)
    outline = 'Filetag     :\t%s\n' % filetag
    ofh.write(outline)
    outline = 'RNG seed    :\t%s\n' % rng_seed
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
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Allele frequency change over time (%s)" % scenario)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Allele frequency")
    x = freq_hist.keys()
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'y', 'm', 'k'])
    markers = itertools.cycle(['o', 's', 'v'])
    for r in xrange(len(recessives)):
        y = []
        for v in freq_hist.values():
            y.append(v[r])
        ax.plot(x, y, color=colors.next(), marker=markers.next(), label=recessives[r][3])
    ax.legend(loc='best')
    filename = "allele_frequency_plot%s.png" % filetag
    plt.savefig(filename, bbox_inches="tight")
    plt.clf()


if __name__ == '__main__':

    # Simulation parameters
    base_bulls = 500        # Initial number of founder bulls in the population
#    base_bulls = 50
    base_cows = 100000      # Initial number of founder cows in the population
#    base_cows = 10000
    base_herds = 200        # Number of herds in the population
#    base_herds = 20
    max_bulls = 500         # Maximum number of live bulls to keep each generation
#    max_bulls = 50
    max_cows = 100000       # Maximum number of live cows to keep each generation
#    max_cows = 10000
    percent = 0.10          # Proportion of bulls to use in the toppct scenario
    generations = 20        # How long to run the simulation
    max_matings = 5000      # The maximum number of matings permitted for each bull (5% of cows)
    debug = True           # Activate (True) or deactivate (False) debugging messages

    # Recessives are stored in a list of lists. The first value in each list
    # is the minor allele frequency in the base population, and the second
    # number is the economic value of the minor allele. If the economic value
    # is $20, that means that the value of an affected homozygote is -$20. The
    # third value in the record indicates if the recessive is lethal (0) or
    # non-lethal (0). The fourth value is a label that is not used for any
    # calculations.
#    default_recessives = [
#        [0.0276, 150, 1, 'Brachyspina'],
#        [0.0192,  40, 1, 'HH1'],
#        [0.0166,  40, 1, 'HH2'],
#        [0.0295,  40, 1, 'HH3'],
#        [0.0037,  40, 1, 'HH4'],
#        [0.0222,  40, 1, 'HH5'],
#        [0.0025, 150, 1, 'BLAD'],
#        [0.0137,  70, 1, 'CVM'],
#        [0.0001,  40, 1, 'DUMPS'],
#        [0.0007, 150, 1, 'Mulefoot'],
#        [0.0071, -20, 0, 'Polled'],
#        [0.0542,  20, 0, 'Red'],
#
#     ]

    with open('recessives.config','r') as inf:
        default_recessives = ast.literal_eval(inf.read())

    # # First, run the random mating scenario
    # print '=' * 80
    # recessives = copy.deepcopy(default_recessives)
    # run_scenario(scenario='random', base_bulls=base_bulls, base_cows=base_cows,
    #              max_bulls=max_bulls, max_cows=max_cows, filetag='_ran_default',
    #              recessives=recessives, rng_seed=None)
    #
    # # Now run truncation selection, just to introduce some genetic trend.
    # print '=' * 80
    # recessives = copy.deepcopy(default_recessives)
    # run_scenario(scenario='toppct', percent=percent, base_bulls=base_bulls,
    #              base_cows=base_cows, max_bulls=max_bulls, max_cows=max_cows,
    #              filetag='_toppct_default', recessives=recessives,
    #              rng_seed=None)
    #
    # # This is the real heart of the analysis, applying Pryce's method, which accounts for
    # # inbreeding but NOT for recessives.
    # print '=' * 80
    # recessives = copy.deepcopy(default_recessives)
    # run_scenario(scenario='pryce', percent=percent, base_bulls=base_bulls, base_cows=base_cows,
    #              base_herds=base_herds, max_bulls=max_bulls, max_cows=max_cows, debug=debug,
    #              filetag='_pryce_default', recessives=recessives, gens=generations,
    #              max_matings=max_matings, rng_seed=None)

    # The 'pryce_r' scenario applies Pryce's method, which accounts for inbreeding
    # and also for recessive effects.
    print '=' * 80
    recessives = copy.deepcopy(default_recessives)
    run_scenario(scenario='pryce_r', percent=percent, base_bulls=base_bulls, base_cows=base_cows,
                 base_herds=base_herds, max_bulls=max_bulls, max_cows=max_cows, debug=debug,
                 filetag='_pryce_r_default', recessives=recessives, gens=generations,
                 max_matings=max_matings, rng_seed=None)