# %%
import enum
import numpy as np
import matplotlib.pyplot as plt
class GA:
    XDIM=2
    DNA_SIZE = 16
    POP_SIZE = 1000
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.003
    N_GENERATIONS = 1000
    X_BOUND = [-5, 5]
    Y_BOUND = [-5, 5]
    Val =[]
    def __init__(Self,*,func,xrange=5,xbits=32 ,xdim=2):
        Self.F=func
        Self.X_BOUND=[-xrange,xrange]
        Self.Y_BOUND=[-xrange,xrange]
        Self.DNA_SIZE=xbits
        Self.XDIM=xdim

    def F(x, y):...

    def get_fitness(Self,pop):
        x,y = Self.translateDNA(pop)
        pred = Self.F(x, y)
        return -(pred - np.max(pred))+1e-3 
    
    def translateDNA(Self,pop): 
        x_pop = pop[:,1::2]
        y_pop = pop[:,::2] 
        x = x_pop.dot(2**np.arange(Self.DNA_SIZE)[::-1])/float(2**Self.DNA_SIZE-1)*(Self.X_BOUND[1]-Self.X_BOUND[0])+Self.X_BOUND[0]
        y = y_pop.dot(2**np.arange(Self.DNA_SIZE)[::-1])/float(2**Self.DNA_SIZE-1)*(Self.Y_BOUND[1]-Self.Y_BOUND[0])+Self.Y_BOUND[0]
        return x,y

    def crossover_and_mutation(Self,pop, CROSSOVER_RATE = 0.8):
        new_pop = []
        for father in pop:		
            child = father		
            if np.random.rand() < CROSSOVER_RATE:			
                mother = pop[np.random.randint(Self.POP_SIZE)]	
                cross_points = np.random.randint(low=0, high=Self.DNA_SIZE*Self.XDIM)	
                child[cross_points:] = mother[cross_points:]		
            Self.mutation(child)	
            new_pop.append(child)
        return new_pop
    
    def mutation(Self,child, MUTATION_RATE=0.003):
        if np.random.rand() < MUTATION_RATE: 				
            mutate_point = np.random.randint(0, Self.DNA_SIZE*Self.XDIM)	
            child[mutate_point] = child[mutate_point]^1 	

    def select(Self,pop, fitness):    
        idx = np.random.choice(np.arange(Self.POP_SIZE), size=Self.POP_SIZE, replace=True,
                           p=(fitness)/(fitness.sum()) )
        return pop[idx]

    def print_info(Self,pop,gen):
        fitness = Self.get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("-----第",gen,"代-----")
        print("max_fitness:", fitness[max_fitness_index])
        x,y = Self.translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
        Z=Self.F(x[max_fitness_index], y[max_fitness_index])
        print("Z:",Z)
        Self.Val.append(Z)

    def plot_3d(Self,r_min=-5,r_max=5,step=0.1):
        xaxis = np.arange(r_min, r_max, step)
        yaxis = np.arange(r_min, r_max, step)
        x, y = np.meshgrid(xaxis, yaxis)
        results1 = Self.F(x, y)
        figure = plt.figure()
        axis = figure.gca( projection='3d')
        axis.plot_surface(x, y, results1, cmap='jet', shade= "false")
        plt.show()
        plt.contour(x,y,results1)
        plt.show()
    def GA(Self,N_GENERATIONS=1000):
        pop = np.random.randint(2, size=(Self.POP_SIZE, Self.DNA_SIZE*2)) 
        for gen in range(N_GENERATIONS):
            x,y = Self.translateDNA(pop)
            pop = np.array(Self.crossover_and_mutation(pop, Self.CROSSOVER_RATE))
            fitness = Self.get_fitness(pop)
            pop = Self.select(pop, fitness) 
            Self.print_info(pop,gen)
        plt.plot(Self.Val)
        plt.show()

def rastrigin(x,y):
    return 20+ x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)
def achley(x,y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
  np.pi * x)+np.cos(2 * np.pi * y))) + np.exp(1) + 20

if __name__ == "__main__":
    np.random.seed(1)
    ga_achley=GA(func=achley,xrange=32.768)
    ga_achley.plot_3d(-32.768,32.768,2.0)
    ga_achley.GA()

    ga_rastrigin=GA(func=rastrigin,xrange=5)
    ga_rastrigin.plot_3d(-5,5,0.1)
    ga_rastrigin.GA()


