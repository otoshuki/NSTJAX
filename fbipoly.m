classdef fbipoly
%FBIPOLY  Polynomial-field library %
%Copyright (rewrite) 2025.  Original algorithm: A. J. Krener
    methods (Static)

        function c = binom(a, b)
        %BINOM  Vectorized generalized binomial coefficient C(a,b).
        %Replaces choose/chuze mex routines

            a = double(a);
            b = double(b);

            %Broadcast A and B to a common size
            common = a + b;
            a = a + zeros(size(common));
            b = b + zeros(size(common));

            c   = ones(size(common));        %b == 0  ->  1
            neg = (b < 0);                   %b <  0  ->  0

            bb = b(~neg);
            if isempty(bb)
                maxb = 0;
            else
                maxb = max(bb(:));
            end

            %One short loop over the degree only (not over elements):
            %at step i, multiply in the factor (a-i+1)/i for every
            %element whose b is still >= i.  After step i, c holds C(a,i)
            for i = 1:maxb
                active = (b >= i);
                c(active) = c(active) .* (a(active) - i + 1) ./ i;
            end

            c(neg) = 0;
            c = round(c);                    %clear floating-point dust
        end

        function m = crd(n, d)
        %CRD  Number of homogeneous monomials of degree d
        %Taken from Krener's implementation
            ns = sum(double(n(:)));
            m  = fbipoly.binom(ns + d - 1, d);
        end

        function m = crdsum(n, d)
        %CRDSUM  Number of monomials of a range of degrees
        %Taken from Krener's implementation
            ns = sum(double(n(:)));
            if isscalar(d)
                d = [d, d];
            end
            d0 = d(1);
            d1 = d(end);
            m  = fbipoly.binom(ns + d1, d1) - fbipoly.binom(ns + d0 - 1, d0 - 1);
        end

        function E = monomials(n, k)
        %MONOMIALS  Exponent table for degree-k monomials in order
            n = double(n);
            k = double(k);

            %degenerate cases
            if k == 0
                E = zeros(1, n);          %the single constant monomial
                return;
            end
            if n == 0
                E = zeros(0, 0);          %no variables, positive degree
                return;
            end
            if n == 1
                E = k;                    %only x0^k
                return;
            end

            %general case
            A = sortrows(nchoosek(1:(n + k - 1), k));   %lex, strictly incr.
            B = A - (0:k - 1);                          %non-decreasing, 1..n

            %Convert each factor list (row of B) to an exponent count row
            M    = size(B, 1);
            rows = repmat((1:M).', 1, k);
            E    = accumarray([rows(:), B(:)], 1, [M, n]);
        end


    end
end
