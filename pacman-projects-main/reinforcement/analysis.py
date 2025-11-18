# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # Crossing the bridge only becomes appealing if the agent can reliably
    # move straight; keeping the discount at 0.9 but driving the transition
    # noise to zero makes slips into the -100 chasms impossible,
    # so the high-reward goal dominates the decision.
    answer_discount = 0.9
    answer_noise = 0
    return answer_discount, answer_noise

def question3a():
    # Low discount keeps the agent focused on quick rewards, and zero noise
    # makes the risky shortcut reliable, so the close exit via the cliff edge
    # becomes the best strategy.
    answer_discount = 0.3
    answer_noise = 0
    answer_living_reward = 0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Adding noise makes the cliff path too dangerous, while a mild negative
    # living reward still nudges the agent to finish promptly, leading it to
    # take the safe top route toward the close exit.
    answer_discount = 0.3
    answer_noise = 0.2
    answer_living_reward = -0.1
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # A higher discount values the larger distant reward, and zero noise lets
    # the agent sprint along the cliff for the fastest access to the +10 exit.
    answer_discount = 0.9
    answer_noise = 0
    answer_living_reward = 0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # The distant exit is still preferred thanks to the high discount, but
    # introducing noise pushes the optimal path away from the cliff toward the
    # upper safe corridor.
    answer_discount = 0.9
    answer_noise = 0.2
    answer_living_reward = 0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # With a living reward larger than any terminal payoff, the agent maximizes
    # return by wandering forever and never touching an exit or the cliff.
    answer_discount = 1
    answer_noise = 0
    answer_living_reward = 1
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question6():
    # Even with epsilon=0 (pure exploitation) the agent never explores the
    # bridge path it has to learn, while any epsilon>0 still causes occasional
    # slips off the bridge within 50 episodes. Those unavoidable random falls
    # keep the success probability under 99% regardless of the learning rate.
    # Therefore no (epsilon, learning_rate) pair can guarantee the optimal
    # policy after only 50 training episodes.
    return 'NOT POSSIBLE'

def question8():
    # Same reasoning as explained in question 6.
    answer_epsilon = None
    answer_learning_rate = None
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
