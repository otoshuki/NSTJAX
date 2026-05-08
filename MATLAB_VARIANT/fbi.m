function [th, la, ff, hh] = fbi(f, h, f_, n, m, p, n_, d)
%FBI  Term-by-term solution of the Francis-Byrnes-Isidori (FBI) PDE for nonlinear regulation
%Author: gpertin, KAIST
%Copyright (rewrite) 2025.  Original algorithm: A. J. Krener

    df = [1, d];                              
    %degree-1 solution
    Fx  = f(:, 1:n);                          %dF/dx        (N x N)
    Fu  = f(:, n+1:n+m);                      %dF/du        (N x M)
    Fxb = f(:, n+m+1:n+m+n_);                 %dF/dx_       (N x N_)
    Hx  = h(:, 1:n);                          %dH/dx        (P x N)
    Hu  = h(:, n+1:n+m);                      %dH/du        (P x M)
    Hxb = h(:, n+m+1:n+m+n_);                 %dH/dx_       (P x N_)
    A_  = f_(:, 1:n_);                        %linear exosystem  x_' = A_ x_

    %Combined linear plant/output map and the selector onto the x-block
    M_  = [Fx, Fu; Hx, Hu];                   %(N+P) x (N+M)
    Sel = [eye(n), zeros(n,m); zeros(p, n+m)];%(N+P) x (N+M), picks the x part

    %Degree 1. Solve, for T (N x N_) and L (M x N_):
    %     Fx*T + Fu*L - T*A_ = -Fxb        (the F rows)
    %     Hx*T + Hu*L        = -Hxb        (the H rows)
    %Using (P*Z*Q) row-stacked = Z row-stacked * kron(P', Q), this is one
    %linear system in the stacked unknown [T;L]
    Op  = kron(M_.',  eye(n_)) - kron(Sel.', A_);
    rhs = -reshape([Fxb; Hxb].', 1, (n+p)*n_);
    sol = rhs * pinv(Op);
    th  = reshape(sol(1, 1:n*n_),            n_, n).';   %N  x N_
    la  = reshape(sol(1, n*n_+1:(n+m)*n_),   n_, m).';   %M  x N_

    %Degrees 2-D. Same linear operator (now acting on degree-k blocks),
    %with a right-hand side assembled from the lower-degree solution
    for k = 2:d
        n_k = fbipoly.crd(n_, k);             %degree-k monomials in x_
        %Linear operator at degree k
        LF = fbipoly.dd(eye(n_k), [n_k, n_], k, A_, [n_, n_], 1, k);
        Op = kron(M_.', eye(n_k)) - kron(Sel.', LF);
        %RHS term 1
        data_hi = fbipoly.prt([f; h], [n,n; p,m; 0,n_], df, [2, k]);
        g_sub   = [th; la; eye(n_), zeros(n_, fbipoly.crdsum(n_, [2, k-1]))];
        comp    = fbipoly.compose(data_hi, [n,n; p,m; 0,n_], [2, k], ...
                                  g_sub,   [n,0; m,0; n_,n_], [1, k-1], k);
        %RHS term 2
        f_hi  = fbipoly.prt(f_, [n_, n_], df, [1, k]);
        deriv = [fbipoly.dd(th, [n, n_], [1, k-1], f_hi, [n_, n_], [1, k], k); ...
                 zeros(p, n_k)];
        rhs = deriv - comp;                           % (N+P) x n_k
        rhs = reshape(rhs.', 1, n_k*(n+p));           % row vector
        %Solve  rhs = sol * Op  (exact if square & nonsingular, else lsq).
        if size(Op,1) == size(Op,2)
            sol = rhs / Op;
        else
            sol = rhs * pinv(Op);
        end
        sol = reshape(sol, n_k, n+m).';               
        th  = [th, sol(1:n,     :)];                  
        la  = [la, sol(n+1:n+m, :)];                  
    end

    %Transverse coordinates  z = x - TH(x_),  v = u - LA(x_),  x_ = x_ .
    ph = [eye(n+m+n_), zeros(n+m+n_, fbipoly.crdsum(n+m+n_, [2, d]))];
    thall = fbipoly.compose(th, [n 0 0; 0 0 n_].', df, ...
                            [zeros(n_, n+m), eye(n_)], [0 0 n_; n m n_].', 1, df);
    ph(1:n, :) = ph(1:n, :) + thall;
    laall = fbipoly.compose(la, [0 m 0; 0 0 n_].', df, ...
                            [zeros(n_, n+m), eye(n_)], [0 0 n_; n m n_].', 1, df);
    ph(n+1:n+m, :) = ph(n+1:n+m, :) + laall;
    nf  = [n 0 0; n m n_].';
    nh  = [p 0 0; n m n_].';
    nth = [n n_];
    nf_ = [n_ n_];
    nph = [n m n_; n m n_].';
    ff = fbipoly.compose(f, nf, df, ph, nph, df, df);
    hh = fbipoly.compose(h, nh, df, ph, nph, df, df);
    temp = fbipoly.dd(th, nth, df, f_, nf_, df, df);
    ff = ff - fbipoly.compose(temp, [n 0 0; 0 0 n_].', df, ...
                              [zeros(n_, n+m), eye(n_)], [0 0 n_; n m n_].', 1, df);
end