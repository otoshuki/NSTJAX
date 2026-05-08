classdef fbipoly
%FBIPOLY  Polynomial-field library %
%Author: gpertin, KAIST
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

        function field = zerofield(nout, nvars, dmax)
        %ZEROFIELD  Empty graded field with room for degrees 0..dmax
            field = struct('nvars', nvars, 'nout', nout, ...
                           'coef', {cell(1, dmax + 1)});
        end

        function field = constfield(nvars)
        %CONSTFIELD  Scalar field equal to the constant 1
            field = struct('nvars', nvars, 'nout', 1, 'coef', {{1}});
        end

        function H = addblock(H, k, B)
        %ADDBLOCK  Add B into the degree-k block of field H
            if k + 1 > numel(H.coef)
                H.coef{k + 1} = [];
            end
            if isempty(H.coef{k + 1})
                H.coef{k + 1} = B;
            else
                H.coef{k + 1} = H.coef{k + 1} + B;
            end
        end

        function s = rowfield(field, i)
        %ROWFIELD  Extract output row i of a field as a scalar (1-output) field
            s = struct('nvars', field.nvars, 'nout', 1, ...
                       'coef', {cell(1, numel(field.coef))});
            for k = 0:numel(field.coef) - 1
                Ck = field.coef{k + 1};
                if ~isempty(Ck)
                    s.coef{k + 1} = Ck(i, :);
                end
            end
        end

        function R = polymul(P, Q, dmax)
        %POLYMUL  Truncated product of two scalar graded polynomials.
        %Different from Krener's approach
            n = P.nvars;
            R = fbipoly.zerofield(1, n, dmax);
            for ip = 0:numel(P.coef) - 1
                Pp = P.coef{ip + 1};
                if isempty(Pp), continue; end
                for iq = 0:numel(Q.coef) - 1
                    Qq = Q.coef{iq + 1};
                    if isempty(Qq), continue; end
                    k = ip + iq;
                    if k > dmax, continue; end
                    Ei = fbipoly.monomials(n, ip);
                    Ej = fbipoly.monomials(n, iq);
                    Mi = size(Ei, 1);
                    Mj = size(Ej, 1);
                    %all monomial pairs (a outer, b inner)
                    Esum  = repelem(Ei, Mj, 1) + repmat(Ej, Mi, 1);
                    cprod = kron(Pp(:), Qq(:));
                    Lk = fbipoly.monomials(n, k);
                    [~, pos] = ismember(Esum, Lk, 'rows');
                    blk = accumarray(pos, cprod, [size(Lk, 1), 1]).';
                    R = fbipoly.addblock(R, k, blk);
                end
            end
        end

        function H = composecore(F, G, dmax)
        %COMPOSECORE  Compose graded fields:  H(y) = F(G(y)), to degree DMAX
            m    = F.nvars;
            n    = G.nvars;
            nout = F.nout;
            if G.nout ~= m
                error('fbipoly:composecore', ...
                    ['COMPOSECORE: F has %d input variables but G has %d ', ...
                     'outputs; they must match.'], m, G.nout);
            end
            %Precompute powers Gpow{i}{e+1} = (i-th output of G)^e, e=0..dmax
            Gpow = cell(1, m);
            for i = 1:m
                Gi = fbipoly.rowfield(G, i);
                pw = cell(1, dmax + 1);
                pw{1} = fbipoly.constfield(n);
                for e = 1:dmax
                    pw{e + 1} = fbipoly.polymul(pw{e}, Gi, dmax);
                end
                Gpow{i} = pw;
            end
            H = fbipoly.zerofield(nout, n, dmax);
            for kf = 0:numel(F.coef) - 1
                Ck = F.coef{kf + 1};
                if isempty(Ck), continue; end
                E = fbipoly.monomials(m, kf);
                for r = 1:size(E, 1)
                    alpha = E(r, :);
                    c     = Ck(:, r);
                    if all(c == 0), continue; end
                    %image = prod_i G_i^{alpha_i}
                    image = fbipoly.constfield(n);
                    for i = 1:m
                        if alpha(i) > 0
                            image = fbipoly.polymul(image, Gpow{i}{alpha(i)+1}, dmax);
                        end
                    end
                    %accumulate c * image into H
                    for kk = 0:numel(image.coef) - 1
                        B = image.coef{kk + 1};
                        if isempty(B), continue; end
                        H = fbipoly.addblock(H, kk, c * B);
                    end
                end
            end
        end

        function [h, nh] = compose(f, nf, df, g, ng, dg, d)
        %COMPOSE  Descriptor-level composition  h = f(g(.))
            if isscalar(df), df = [df df]; end
            if isscalar(dg), dg = [dg dg]; end
            if isscalar(d),  d  = [d  d ]; end
            nf = double(nf);
            ng = double(ng);
            if size(nf, 1) ~= size(ng, 1)
                error('fbipoly:compose', 'NF and NG must have the same number of rows.');
            end
            finlast = nf(:, end);
            goutcol = ng(:, 1);
            %Require full clean alignment
            if ~isequal(finlast, goutcol)
                error('fbipoly:compose:alignment', ...
                    ['COMPOSE: f-input subvectors (last column of NF) must ', ...
                     'equal g-output subvectors (first column of NG) row by ', ...
                     'row.  Partial substitution is not supported.']);
            end
            F = fbipoly.decode(f, nf, df);
            G = fbipoly.decode(g, ng, dg);
            H = fbipoly.composecore(F, G, d(end));
            nh = nf;
            nh(:, end) = ng(:, end);
            h = fbipoly.encode(H, nh, d);
        end

        function A = addfield(A, B, dmax)
        %ADDFIELD  Add field B into field A, keeping degrees 0..DMAX
            for k = 0:numel(B.coef) - 1
                Bk = B.coef{k + 1};
                if isempty(Bk) || k > dmax, continue; end
                A = fbipoly.addblock(A, k, Bk);
            end
        end

        function dP = partialfield(P, j)
        %PARTIALFIELD  Partial derivative dP/dx_j of a (multi-output) field.
            ns   = P.nvars;
            nout = P.nout;
            dP   = fbipoly.zerofield(nout, ns, max(0, numel(P.coef) - 2));
            for k = 1:numel(P.coef) - 1
                Ck = P.coef{k + 1};
                if isempty(Ck), continue; end
                E    = fbipoly.monomials(ns, k);
                aj   = E(:, j);
                keep = aj > 0;
                if ~any(keep), continue; end
                Eb        = E(keep, :);
                Eb(:, j)  = Eb(:, j) - 1;
                scale     = aj(keep).';
                Cs        = Ck(:, keep) .* scale;
                Lk1       = fbipoly.monomials(ns, k - 1);
                [~, pos]  = ismember(Eb, Lk1, 'rows');
                Mk1       = size(Lk1, 1);
                ridx      = repmat((1:nout).', 1, numel(pos));
                cidx      = repmat(pos(:).', nout, 1);
                blk       = accumarray([ridx(:), cidx(:)], Cs(:), [nout, Mk1]);
                dP        = fbipoly.addblock(dP, k - 1, blk);
            end
        end

        function R = polymulvec(P, Q, dmax)
        %POLYMULVEC  Product of a multi-output field P and a scalar field Q.
            n    = P.nvars;
            nout = P.nout;
            R    = fbipoly.zerofield(nout, n, dmax);

            for ip = 0:numel(P.coef) - 1
                Pp = P.coef{ip + 1};
                if isempty(Pp), continue; end
                for iq = 0:numel(Q.coef) - 1
                    Qq = Q.coef{iq + 1};
                    if isempty(Qq), continue; end
                    k = ip + iq;
                    if k > dmax, continue; end
                    Ei = fbipoly.monomials(n, ip);
                    Ej = fbipoly.monomials(n, iq);
                    Mi = size(Ei, 1);
                    Mj = size(Ej, 1);
                    Esum = repelem(Ei, Mj, 1) + repmat(Ej, Mi, 1);
                    Pexp = Pp(:, repelem(1:Mi, Mj));
                    Qexp = Qq(repmat(1:Mj, 1, Mi));
                    cpair = Pexp .* Qexp;
                    Lk = fbipoly.monomials(n, k);
                    [~, pos] = ismember(Esum, Lk, 'rows');
                    Mk   = size(Lk, 1);
                    K    = numel(pos);
                    ridx = repmat((1:nout).', 1, K);
                    cidx = repmat(pos(:).', nout, 1);
                    blk  = accumarray([ridx(:), cidx(:)], cpair(:), [nout, Mk]);
                    R = fbipoly.addblock(R, k, blk);
                end
            end
        end

        function H = ddmul(F, G, dmax)
        %DDMUL  Directional derivative  H = (dF/dx) * G = sum_j dF/dx_j * G_j

            ns   = F.nvars;
            nout = F.nout;
            if G.nvars ~= ns || G.nout ~= ns
                error('fbipoly:ddmul', ...
                    ['DDMUL needs G with %d inputs and %d outputs (to match ', ...
                     'F''s %d variables); got %d inputs, %d outputs.'], ...
                     ns, ns, ns, G.nvars, G.nout);
            end

            H = fbipoly.zerofield(nout, ns, dmax);
            for j = 1:ns
                dFj = fbipoly.partialfield(F, j);
                Gj  = fbipoly.rowfield(G, j);
                Pj  = fbipoly.polymulvec(dFj, Gj, dmax);
                H   = fbipoly.addfield(H, Pj, dmax);
            end
        end

        function [h, nh] = dd(f, nf, df, g, ng, dg, d)
        %DD  Descriptor-level directional derivative
            if isscalar(df), df = [df df]; end
            if isscalar(dg), dg = [dg dg]; end
            if isscalar(d),  d  = [d  d ]; end
            nf = double(nf);
            ng = double(ng);
            if size(ng, 2) ~= 2
                error('fbipoly:dd', 'NG must have exactly 2 columns.');
            end
            if size(nf, 1) ~= size(ng, 1)
                error('fbipoly:dd', 'NF and NG must have the same number of rows.');
            end
            if ~isequal(nf(:, end), ng(:, 1)) || ~isequal(nf(:, end), ng(:, 2))
                error('fbipoly:dd:alignment', ...
                    ['DD supports only the case where f-inputs, g-outputs and ', ...
                     'g-inputs coincide (as used throughout FBI).']);
            end
            F = fbipoly.decode(f, nf, df);
            G = fbipoly.decode(g, ng, dg);
            H = fbipoly.ddmul(F, G, d(end));
            nh = nf;
            h  = fbipoly.encode(H, nh, d);
        end

        function h = prt(f, nf, df, d)
        %PRT  Extract the degrees-D part of a packed field
            if isscalar(df), df = [df df]; end
            if isscalar(d),  d  = [d  d ]; end
            field = fbipoly.decode(f, nf, df);
            h     = fbipoly.encode(field, nf, d);
        end

    end
end
