
import java.util.*;
import java.util.function.Supplier;

/**
 * 1D diffusion solver (explicit FTCS) + simple evolutionary algorithm to evolve parameters D and dt.
 */
public class DiffusionEA {

	// Problem domain and discretization
	static final double L = 1.0;
	static final int NX = 101;           // spatial points
	static final double DX = L / (NX - 1);
	static final double T_FINAL = 0.1;   // final time to compare
	static final int MAX_STEPS = 2000000; // safety cap

	// EA parameters
	static final int POP = 50;
	static final int GENERATIONS = 150;
	static final double MUTATION_STD = 0.1; // relative mutation scale
	static final double CROSSOVER_RATE = 0.9;
	static final int TOURNAMENT = 3;
	static final int ELITE = 2;

	// Parameter bounds (D, dt)
	static final double D_MIN = 1e-4;
	static final double D_MAX = 1.0;
	static final double DT_MIN = 1e-6;
	static final double DT_MAX = 1e-2;

	static final Random rnd = new Random(42);

	// Represents candidate: genome[0]=D, genome[1]=dt
	static class Individual {
		double[] genome = new double[2];
		double fitness = Double.POSITIVE_INFINITY;
	}

	// Create initial condition: e.g., Gaussian bump in center
	static double[] initialCondition() {
		double[] u0 = new double[NX];
		double center = 0.5 * L;
		double sigma = 0.05;
		for (int i = 0; i < NX; i++) {
			double x = i * DX;
			u0[i] = Math.exp(-Math.pow(x - center, 2) / (2 * sigma * sigma));
		}
		return u0;
	}

	// Create a target profile. For demonstration, use exact solution for chosen D_true (or a slightly perturbed initial condition).
	static double[] targetProfile(double D_true, double dtSim) {
		// Use solver to compute target at T_FINAL using stable dtSim
		return runExplicitSolver(initialCondition(), D_true, dtSim, T_FINAL);
	}

	// Explicit FTCS solver for 1D diffusion with Dirichlet boundaries (u=0 at boundaries)
	static double[] runExplicitSolver(double[] u0, double D, double dt, double tFinal) {
		double r = D * dt / (DX * DX);
		// Enforce stability: if unstable, return large-error signal by returning null
		if (r > 0.5) return null;

		double[] u = Arrays.copyOf(u0, u0.length);
		double[] uNew = new double[NX];
		int steps = (int) Math.ceil(tFinal / dt);
		if (steps > MAX_STEPS) steps = MAX_STEPS;

		for (int n = 0; n < steps; n++) {
			// interior points
			for (int i = 1; i < NX - 1; i++) {
				uNew[i] = u[i] + r * (u[i - 1] - 2 * u[i] + u[i + 1]);
			}
			// Dirichlet boundaries (zero)
			uNew[0] = 0.0;
			uNew[NX - 1] = 0.0;
			// swap
			double[] tmp = u;
			u = uNew;
			uNew = tmp;
		}
		return u;
	}

	// MSE between two profiles. If either is null, return huge penalty.
	static double mse(double[] a, double[] b) {
		if (a == null || b == null) return 1e6;
		double s = 0.0;
		for (int i = 0; i < a.length; i++) {
			double d = a[i] - b[i];
			s += d * d;
		}
		return s / a.length;
	}

	// Random initialization within bounds (log-uniform for D might be better, but keep uniform)
	static Individual randomIndividual() {
		Individual ind = new Individual();
		ind.genome[0] = D_MIN + rnd.nextDouble() * (D_MAX - D_MIN);
		ind.genome[1] = DT_MIN + rnd.nextDouble() * (DT_MAX - DT_MIN);
		return ind;
	}

	// Evaluate individual's fitness: MSE to target profile
	static void evaluate(Individual ind, double[] target) {
		// Ensure dt leads to stable r; if not, apply heavy penalty
		double[] sim = runExplicitSolver(initialCondition(), ind.genome[0], ind.genome[1], T_FINAL);
		ind.fitness = mse(sim, target);
	}

	// Tournament selection
	static Individual tournamentSelect(List<Individual> pop) {
		Individual best = null;
		for (int i = 0; i < TOURNAMENT; i++) {
			Individual c = pop.get(rnd.nextInt(pop.size()));
			if (best == null || c.fitness < best.fitness) best = c;
		}
		return best;
	}

	// Uniform crossover
	static Individual crossover(Individual a, Individual b) {
		Individual child = new Individual();
		for (int i = 0; i < child.genome.length; i++) {
			child.genome[i] = (rnd.nextDouble() < 0.5) ? a.genome[i] : b.genome[i];
		}
		return child;
	}

	// Gaussian mutation (relative)
	static void mutate(Individual ind) {
		for (int i = 0; i < ind.genome.length; i++) {
			if (rnd.nextDouble() < 0.5) {
				double val = ind.genome[i];
				double std = MUTATION_STD * Math.max(Math.abs(val), 1e-8);
				val += rnd.nextGaussian() * std;
				// clamp to bounds
				if (i == 0) val = clamp(val, D_MIN, D_MAX);
				else val = clamp(val, DT_MIN, DT_MAX);
				ind.genome[i] = val;
			}
		}
	}

	static double clamp(double v, double lo, double hi) {
		return Math.max(lo, Math.min(hi, v));
	}

	public static void main(String[] args) {
		// Create target using a true parameter (unknown to EA)
		double D_true = 0.01;
		double dtTarget = 5e-5; // small, stable
		double[] target = targetProfile(D_true, dtTarget);

		// Initialize population
		List<Individual> population = new ArrayList<>();
		for (int i = 0; i < POP; i++) {
			population.add(randomIndividual());
		}

		// Evaluate initial population
		for (Individual ind : population) evaluate(ind, target);

		// Evolution loop
		for (int gen = 0; gen < GENERATIONS; gen++) {
			// sort by fitness
			population.sort(Comparator.comparingDouble(i -> i.fitness));

			// Logging best each generation (simple)
			Individual best = population.get(0);
			System.out.printf("Gen %03d: best fitness=%.6e D=%.6f dt=%.6e%n",
							  gen, best.fitness, best.genome[0], best.genome[1]);

			// Create next gen with elitism
			List<Individual> next = new ArrayList<>();
			for (int e = 0; e < ELITE; e++) next.add(population.get(e));

			// Fill rest
			while (next.size() < POP) {
				Individual p1 = tournamentSelect(population);
				Individual child;
				if (rnd.nextDouble() < CROSSOVER_RATE) {
					Individual p2 = tournamentSelect(population);
					child = crossover(p1, p2);
				} else {
					// clone
					child = new Individual();
					child.genome = Arrays.copyOf(p1.genome, p1.genome.length);
				}
				mutate(child);
				evaluate(child, target);
				next.add(child);
			}

			population = next;

			// Optional early stop
			if (population.get(0).fitness < 1e-8) break;
		}

		// Final best
		population.sort(Comparator.comparingDouble(i -> i.fitness));
		Individual best = population.get(0);
		System.out.println("=== Final Best ===");
		System.out.printf("fitness=%.6e D=%.6f dt=%.6e%n", best.fitness, best.genome[0], best.genome[1]);

		// Show comparison of profiles (print subset)
		double[] simBest = runExplicitSolver(initialCondition(), best.genome[0], best.genome[1], T_FINAL);
		System.out.println("x u_target u_best");
		for (int i = 0; i < NX; i += 10) {
			double x = i * DX;
			System.out.printf("%.3f %.6f %.6f%n",
							  x, target[i], simBest == null ? Double.NaN : simBest[i]);
		}
	}
}