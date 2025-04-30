import numpy as np
from fcmeans import FCM

def bbo_optimize(init_pop, cost_func, num_iters=10):
    pop = init_pop.copy()
    best_solution = pop[0]
    best_score = cost_func(best_solution)

    for _ in range(num_iters):
        for i in range(len(pop)):
            mutated = pop[i] + np.random.normal(0, 0.1, pop[i].shape)
            mutated = np.clip(mutated, 0, 1)
            score = cost_func(mutated)
            if score < best_score:
                best_score = score
                best_solution = mutated
    return best_solution

def bbo_fcm_clustering(data, n_clusters=3):
    def cost(center_weights):
        fcm = FCM(n_clusters=n_clusters)
        fcm.centers = center_weights.reshape((n_clusters, -1))
        fcm.fit(data)
        return -fcm._membership(data).mean()

    init_centers = np.random.rand(n_clusters, data.shape[1])
    optimized_centers = bbo_optimize([init_centers], cost_func=cost, num_iters=5)

    fcm = FCM(n_clusters=n_clusters)
    fcm.centers = optimized_centers.reshape((n_clusters, -1))
    fcm.fit(data)
    return fcm.predict(data)
