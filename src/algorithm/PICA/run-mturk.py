import argparse
import numpy as np
from numpy.random import choice, uniform, permutation

from Dataset import *
from EM import *

# Run multiple trials of a test
if __name__=='__main__':
  np.random.seed()

  parser = argparse.ArgumentParser(description='Run trials of a test.')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-n', action='store_true', help='Runs the naive algorithm outlined in our paper')
  parser.add_argument('-m', action='store_true', help='Computes Majority Vote')
  args = parser.parse_args()

  accuracies = []
  cross_entropies = []


  for sim in range(100): # Run 100 simulations
    data = init_from_file("../../../data/clustering_MechanicalTurk/%d.txt" % sim, 1, not args.r, True)

    EM(data)
    acc, ce = data.permutedAcc()
    result = "Simulation %d: %.2f %% | %.2f CE\n" % (sim, acc*100, ce)
    cross_entropies.append(ce)
    
    print result
    # fp.write(result)
    accuracies.append(acc)

  average_acc = sum(accuracies) / len(accuracies)
  acc_std = np.std(accuracies)
  result = "\nAverage: %.2f +- %.2f %% \n" % (average_acc * 100, acc_std * 100)
  # if args.m or args.n:
  #   result = "\nAverage: %.2f %% \n" % (average_acc * 100)
  #
  # else:
  #   average_ce = sum(cross_entropies) / len(cross_entropies)
  #   result = "\nAverage: %.2f %% | %.2f CE\n" % (average_acc, average_ce)
  
  print result
  # fp.write(result)
  # fp.close()