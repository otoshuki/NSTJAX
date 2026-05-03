%Author: gpertin, KAIST

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

            c   = ones(size(common));
            neg = (b < 0);

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
            c = round(c);
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
            %constant monomial
            if k == 0
                E = zeros(1, n);
                return;
            end
            %no variables, positive degree
            if n == 0
                E = zeros(0, 0);
                return;
            end
            %only x0^k
            if n == 1
                E = k;
                return;
            end

            %general case
            A = sortrows(nchoosek(1:(n + k - 1), k));
            B = A - (0:k - 1);

            %Convert each factor list (row of B) to an exponent count row
            M    = size(B, 1);
            rows = repmat((1:M).', 1, k);
            E    = accumarray([rows(:), B(:)], 1, [M, n]);
        end

        function [nout, ns, svlth] = parsedesc(nf)
        %PARSEDESC  Interpret a toolbox size descriptor NF
            nf = double(nf);
            nc = size(nf, 2);
            if nc < 2
                error('fbipoly:parsedesc', ...
                      'Descriptor NF must have at least 2 columns.');
            end
            colsums = sum(nf, 1);
            nout    = prod(colsums(1:nc-1));
            lastcol = nf(:, nc);
            ns      = sum(lastcol);
            svlth   = lastcol(lastcol > 0).';
        end

        function E = redmonomials(svlth, k)
        %REDMONOMIALS  Degree-k exponent rows in the toolbox REDUCED order
        %Following Krener's implementation
            svlth = svlth(svlth > 0);
            ns    = sum(svlth);
            E     = fbipoly.monomials(ns, k);
            if size(E, 1) <= 1
                return;
            end
            %Sort key reproducing the reduced ordering
            key = zeros(size(E, 1), 0);
            c   = 0;
            for i = 1:numel(svlth)
                li    = svlth(i);
                block = E(:, c+1:c+li);
                key   = [key, -sum(block, 2), -block];
                c     = c + li;
            end
            [~, ord] = sortrows(key);
            E = E(ord, :);
        end

        function blk = fieldblock(field, k)
        %FIELDBLOCK  Degree-k coefficient block of a field (lex order)
            w = fbipoly.crd(field.nvars, k);
            if k + 1 <= numel(field.coef) && ~isempty(field.coef{k + 1})
                blk = field.coef{k + 1};
            else
                blk = zeros(field.nout, w);
            end
        end

        function field = decode(f, nf, drange)
        %DECODE  Packed toolbox matrix  ->  graded internal field
        %Following Krener's implementation
            if isscalar(drange), drange = [drange, drange]; end
            d0 = drange(1);
            d1 = drange(end);
            [ndesc, ns, svlth] = fbipoly.parsedesc(nf);
            nout = size(f, 1);
            if ndesc ~= nout
                error('fbipoly:decode:nout', ...
                    ['DECODE: descriptor implies %d output rows but F has ', ...
                     '%d.'], ndesc, nout);
            end
            want = fbipoly.crdsum(ns, [d0 d1]);
            if size(f, 2) ~= want
                error('fbipoly:decode:size', ...
                    ['DECODE: F has %d columns but %d are expected for ', ...
                     'ns=%d, degrees %d..%d.'], size(f, 2), want, ns, d0, d1);
            end
            coef = cell(1, d1 + 1);
            col  = 0;
            for k = d0:d1
                w     = fbipoly.crd(ns, k);
                block = f(:, col+1:col+w);
                col   = col + w;

                L = fbipoly.monomials(ns, k);
                R = fbipoly.redmonomials(svlth, k);
                [~, pos] = ismember(L, R, 'rows');
                coef{k + 1} = block(:, pos);
            end
            field = struct('nvars', ns, 'nout', nout, 'coef', {coef});
        end

        function f = encode(field, nf, drange)
        %ENCODE  Graded internal field  ->  packed toolbox matrix
        %Following Krener's implementation
            if isscalar(drange), drange = [drange, drange]; end
            d0 = drange(1);
            d1 = drange(end);
            [~, ns, svlth] = fbipoly.parsedesc(nf);
            if ns ~= field.nvars
                error('fbipoly:encode:nvars', ...
                    ['ENCODE: descriptor implies %d variables but FIELD has ', ...
                     '%d.'], ns, field.nvars);
            end
            blocks = cell(1, d1 - d0 + 1);
            for k = d0:d1
                blk = fbipoly.fieldblock(field, k);
                L = fbipoly.monomials(ns, k);
                R = fbipoly.redmonomials(svlth, k);
                [~, pos] = ismember(R, L, 'rows');
                blocks{k - d0 + 1} = blk(:, pos);
            end
            f = [blocks{:}];
            if isempty(f)
                f = zeros(field.nout, 0);
            end
        end

        function y = evalfield(field, x)
        %EVALFIELD  Evaluate a graded field at a point (testing helper)
            x     = x(:).';
            nvars = field.nvars;
            nout  = field.nout;
            y     = zeros(nout, 1);
            for k = 0:numel(field.coef) - 1
                Ck = field.coef{k + 1};
                if isempty(Ck), continue; end
                E  = fbipoly.monomials(nvars, k);
                mv = prod(x .^ E, 2);
                y  = y + Ck * mv;
            end
        end
    end
end
