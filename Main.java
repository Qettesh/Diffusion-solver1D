public class Main{a
		// Example run
    public static void main(String[] args) {
        double[] alpha = buildAlphaGrid();
        double[] u0 = initialCondition();
        double dt = DT;
        double tFinal = T_FINAL;

        double[] uFinal = crankNicolson(u0, alpha, dt, tFinal);
        if (uFinal == null) {
            System.err.println("Solver failed.");
            return;
        }

        System.out.println("Solution at final time:");
        for (int i = 0; i < NX; i += 20) {
            double x = i * DX;
            System.out.printf("x=%.3f u=%.6e alpha=%.6e%n", x, uFinal[i], alpha[i]);
        }
}
