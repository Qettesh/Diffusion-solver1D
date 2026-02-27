import java.util.*;
import java.util.function.DoubleUnaryOperator;

/**
 * Crank-Nicolson solver for 1D nonlinear diffusion:
 *   u_t = ∂x( D(u,x) ∂x u ),  D(u,x) = D0 * (1 + alpha(x) * u )
 *
 * Dirichlet BCs: u(0)=u(L)=0
 */
public class CNNonlinearDiffusion {
    // domain
    static final double L = 1.0;
    static final int NX = 201;
    static final double DX = L / (NX - 1);

    // time
    static final double T_FINAL = 0.05;
    static final double DT = 5e-4; // choose based on stability / accuracy

    // Newton
    static final int NEWTON_MAX_IT = 30;
    static final double NEWTON_TOL = 1e-8;

    static final double D0 = 0.02;

    // Example alpha(x): array of length NX (can be spatially varying)
    static double[] buildAlphaGrid() {
        double[] alpha = new double[NX];
        for (int i = 0; i < NX; i++) {
            double x = i * DX;
            // example: bump near center
            alpha[i] = 2.0 * Math.exp(-Math.pow(x - 0.5, 2) / (2 * 0.08 * 0.08));
        }
        return alpha;
    }

    // initial condition (Gaussian bump)
    static double[] initialCondition() {
        double[] u0 = new double[NX];
        double center = 0.5 * L;
        double sigma = 0.06;
        for (int i = 0; i < NX; i++) {
            double x = i * DX;
            u0[i] = Math.exp(-Math.pow(x - center, 2) / (2 * sigma * sigma));
        }
        // enforce Dirichlet
        u0[0] = 0.0;
        u0[NX - 1] = 0.0;
        return u0;
    }

    // D(u,x) = D0 * (1 + alpha(x) * u)
    static double Dof(double u, double alpha) {
        return D0 * (1.0 + alpha * u);
    }

    // derivative dD/du = D0 * alpha
    static double dDdu(double alpha) {
        return D0 * alpha;
    }

    // Main CN time stepping driver
    static double[] crankNicolson(double[] u0, double[] alpha, double dt, double tFinal) {
        int steps = (int) Math.ceil(tFinal / dt);
        double[] u = Arrays.copyOf(u0, u0.length);
        double[] uNew = new double[NX];

        // allocate arrays for Newton: residual and tridiagonal Jacobian (a: lower, b: diag, c: upper)
        double[] R = new double[NX];
        double[] a = new double[NX]; // lower diag (a[1..N-1], a[0] unused)
        double[] b = new double[NX]; // main diag
        double[] c = new double[NX]; // upper diag (c[0..N-2], c[N-1] unused)
        double[] delta = new double[NX];

        for (int n = 0; n < steps; n++) {
            // time level n: u, unknown u^{n+1} to solve for
            // initial guess: take previous time step (implicit Euler / good initial guess)
            System.arraycopy(u, 0, uNew, 0, NX);
            // enforce Dirichlet boundaries
            uNew[0] = 0.0;
            uNew[NX - 1] = 0.0;

            // Precompute explicit fluxes at time n (for RHS)
            double[] Fn = computeFluxes(u, alpha);

            // Newton iterations to solve CN nonlinear system
            boolean converged = false;
            for (int it = 0; it < NEWTON_MAX_IT; it++) {
                // Build residual R and Jacobian entries a,b,c for interior nodes i=1..NX-2
                // For CN: residual for interior i:
                // R_i = u_i^{n+1} - u_i^n - (dt/2dx) * [F_{i+1/2}^{n+1} - F_{i-1/2}^{n+1} + F_{i+1/2}^n - F_{i-1/2}^n]
                // where F_{i+1/2}^{n+1} = - D( (u_i^{n+1}+u_{i+1}^{n+1})/2 , x ) * (u_{i+1}^{n+1} - u_i^{n+1}) / dx

                Arrays.fill(a, 0.0);
                Arrays.fill(b, 0.0);
                Arrays.fill(c, 0.0);
                Arrays.fill(R, 0.0);

                for (int i = 1; i < NX - 1; i++) {
                    // compute F^{n+1} at faces involving uNew
                    // face i+1/2 depends on uNew[i], uNew[i+1]
                    double sPlus = 0.5 * (uNew[i] + uNew[i + 1]);
                    double gradPlus = (uNew[i + 1] - uNew[i]) / DX;
                    double alphaPlus = 0.5 * (alpha[i] + alpha[i + 1]);
                    double Dplus = Dof(sPlus, alphaPlus);

                    double sMinus = 0.5 * (uNew[i - 1] + uNew[i]);
                    double gradMinus = (uNew[i] - uNew[i - 1]) / DX;
                    double alphaMinus = 0.5 * (alpha[i - 1] + alpha[i]);
                    double Dminus = Dof(sMinus, alphaMinus);

                    double Fnp1_plus = -Dplus * gradPlus;
                    double Fnp1_minus = -Dminus * gradMinus;

                    double Fn_plus = Fn[i];     // Fn[i] stored as F_{i+1/2} at index i
                    double Fn_minus = Fn[i - 1];

                    // Residual
                    R[i] = uNew[i] - u[i] - (dt / (2.0 * DX)) * ( (Fnp1_plus - Fnp1_minus) + (Fn_plus - Fn_minus) );

                    // Now Jacobian: derivatives of R[i] wrt uNew[i-1], uNew[i], uNew[i+1]
                    // Compute partials of F_{i+1/2} and F_{i-1/2}
                    // For face i+1/2: s = 0.5(u_i + u_{i+1}), g = (u_{i+1}-u_i)/DX, D = D(s)
                    // F = -D * g
                    // dF/du_{i+1} = -[ D'(s)*(1/2)*g + D*(1/DX) ]
                    // dF/du_i     = -[ D'(s)*(1/2)*g - D*(1/DX) ]
                    double Dprime_plus = dDdu(alphaPlus);
                    double gPlus = gradPlus;
                    double dFplus_dip1 = - ( Dprime_plus * 0.5 * gPlus + Dplus * (1.0 / DX) );
                    double dFplus_di   = - ( Dprime_plus * 0.5 * gPlus - Dplus * (1.0 / DX) );

                    double Dprime_minus = dDdu(alphaMinus);
                    double gMinus = gradMinus;
                    double dFminus_di   = - ( Dprime_minus * 0.5 * gMinus + Dminus * (1.0 / DX) );
                    double dFminus_dim1 = - ( Dprime_minus * 0.5 * gMinus - Dminus * (1.0 / DX) );

                    // Contributions to Jacobian entries (factor dt/(2 DX) * sign)
                    double coeff = dt / (2.0 * DX);

                    // dR/du_{i-1}: only via F_{i-1/2} (Fnp1_minus)
                    a[i] = - coeff * ( - dFminus_dim1 ); // minus because residual has - (Fnp1_plus - Fnp1_minus)
                    // Explanation: R contains -coeff*(Fnp1_plus - Fnp1_minus) => derivative wrt u_{i-1} is -coeff*( -dFminus/du_{i-1} ) = coeff * dFminus/du_{i-1}
                    // we compute dFminus_dim1 = dF_{i-1/2}/du_{i-1}, so a[i] = coeff * dFminus_dim1
                    a[i] = coeff * dFminus_dim1;

                    // dR/du_{i+1}: only via F_{i+1/2}
                    c[i] = - coeff * ( dFplus_dip1 ); // derivative of -coeff*(Fnp1_plus - ...) wrt u_{i+1} is -coeff * dFplus/du_{i+1}
                    // keep sign consistent

                    // dR/du_i: from uNew[i] term (1), and from both faces
                    double diag = 1.0;
                    // contribution from F_{i+1/2}: -coeff * dFplus/du_i
                    diag += - coeff * dFplus_di;
                    // contribution from F_{i-1/2}: -coeff * ( - dFminus_di ) = coeff * dFminus_di
                    diag += coeff * dFminus_di;
                    b[i] = diag;
                }

                // Boundary rows (Dirichlet) set to enforce uNew[0]=0, uNew[NX-1]=0
                b[0] = 1.0; a[0] = 0.0; c[0] = 0.0; R[0] = uNew[0] - 0.0;
                b[NX - 1] = 1.0; a[NX - 1] = 0.0; c[NX - 1] = 0.0; R[NX - 1] = uNew[NX - 1] - 0.0;

                // Solve linear system J * delta = -R (tridiagonal)
                for (int i = 0; i < NX; i++) delta[i] = 0.0;
                boolean solved = solveTridiagonal(a, b, c, negate(R), delta);
                if (!solved) {
                    System.err.println("Tridiagonal solve failed at time step " + n + ", Newton iter " + it);
                    return null;
                }

                // line-search / damping to ensure residual reduces
                double lambda = 1.0;
                double[] uTrial = new double[NX];
                double Rnorm = norm2(R);
                boolean accepted = false;
                for (int ls = 0; ls < 10; ls++) {
                    for (int i = 0; i < NX; i++) uTrial[i] = uNew[i] + lambda * delta[i];
                    // enforce Dirichlet
                    uTrial[0] = 0.0; uTrial[NX - 1] = 0.0;
                    // compute residual at trial
                    double[] Ftrial = computeFluxes(uTrial, alpha);
                    double[] Rtrial = new double[NX];
                    for (int i = 1; i < NX - 1; i++) {
                        double Fp = -Dof(0.5*(uTrial[i]+uTrial[i+1]), 0.5*(alpha[i]+alpha[i+1])) * ((uTrial[i+1]-uTrial[i])/DX);
                        double Fm = -Dof(0.5*(uTrial[i-1]+uTrial[i]), 0.5*(alpha[i-1]+alpha[i])) * ((uTrial[i]-uTrial[i-1])/DX);
                        Rtrial[i] = uTrial[i] - u[i] - (dt / (2.0 * DX)) * ( (Fp - Fm) + (Fn[i] - Fn[i-1]) );
                    }
                    Rtrial[0] = uTrial[0];
                    Rtrial[NX - 1] = uTrial[NX - 1];
                    double RtrialNorm = norm2(Rtrial);
                    if (RtrialNorm < Rnorm || lambda < 1e-4) {
                        // accept
                        System.arraycopy(uTrial, 0, uNew, 0, NX);
                        accepted = true;
                        break;
                    }
                    lambda *= 0.5;
                }
                if (!accepted) {
                    // cannot find acceptable step, abort Newton
                    System.err.println("Newton line search failed at time step " + n);
                    return null;
                }

                // check convergence: norm of delta small relative
                double dnorm = norm2(delta);
                if (dnorm < NEWTON_TOL) {
                    converged = true;
                    break;
                }
            } // end Newton

            if (!converged) {
                System.err.println("Newton did not converge at time step " + n);
                return null;
            }

            // accept step
            System.arraycopy(uNew, 0, u, 0, NX);
        } // end time steps

        return u;
    }

    // Compute fluxes F_{i+1/2} at index i (i=0..NX-2) for given u and alpha
    static double[] computeFluxes(double[] u, double[] alpha) {
        double[] F = new double[NX - 1];
        for (int i = 0; i < NX - 1; i++) {
            double s = 0.5 * (u[i] + u[i + 1]);
            double alph = 0.5 * (alpha[i] + alpha[i + 1]);
            double D = Dof(s, alph);
            double grad = (u[i + 1] - u[i]) / DX;
            F[i] = -D * grad;
        }
        return F;
    }

    // Tridiagonal solver (Thomas). a[0] unused (or zero), c[N-1] unused.
    // Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    static boolean solveTridiagonal(double[] a, double[] b, double[] c, double[] d, double[] x) {
        int n = b.length;
        double[] cp = new double[n];
        double[] dp = new double[n];

        if (Math.abs(b[0]) < 1e-16) return false;
        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];

        for (int i = 1; i < n; i++) {
            double denom = b[i] - a[i] * cp[i - 1];
            if (Math.abs(denom) < 1e-16) return false;
            cp[i] = (i == n - 1) ? 0.0 : c[i] / denom;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
        }

        x[n - 1] = dp[n - 1];
        for (int i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
        return true;
    }

    static double[] negate(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = -v[i];
        return r;
    }

    static double norm2(double[] v) {
        double s = 0.0;
        for (double w : v) s += w * w;
        return Math.sqrt(s);
    }

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
}
