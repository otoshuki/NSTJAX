classdef fbipoly
%FBIPOLY  Polynomial-field library %
%Copyright (rewrite) 2025.  Original algorithm: A. J. Krener.
    methods (Static)

        function c = binom(a, b)
        %BINOM  Vectorized generalized binomial coefficient C(a,b).
        %Replaces choose/chuze mex routines

            a = double(a);
            b = double(b);

            %Broadcast A and B to a common size.
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
            %element whose b is still >= i.  After step i, c holds C(a,i).
            for i = 1:maxb
                active = (b >= i);
                c(active) = c(active) .* (a(active) - i + 1) ./ i;
            end

            c(neg) = 0;
            c = round(c);                    %clear floating-point dust
        end

    end
end
