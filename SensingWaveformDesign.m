%% Sensing Waveform Design
M = 32;                 % length of waveform
%% -------------------- correlation operators --------------------
J = @(m) diag(ones(M-abs(m),1), m);
lags = -(M-1):(M-1);
S = lags(lags~=0);
J0 = J(0);
Jk = cell(numel(S),1); 
JkS = cell(numel(S),1); 
for i=1:numel(S)
    Jk{i}=J(S(i)); 
    JkS{i} = sparse(J(S(i)));
end

%% -------------------- Proposed --------------------
n = (0:M-1).';
z = exp(-1j*pi*1*(n.^2)/M);
R = z*z';   % warm start
rho = 1;                          % initial DC (auto-adapt)
epsR = 1e-6;
for a_it = 1:200
    [V,D] = eig((R+R')/2);
    [lam,ord] = sort(real(diag(D)),'descend');
    u = V(:,ord(1)); 
    q = trace(J0*R) / ( epsR + sum_cell_sqabs(Jk,R) );
    cvx_begin sdp quiet
            variable Rv(M,M) hermitian semidefinite
            expressions tk(numel(Jk))
            for j=1:numel(Jk)
                tk(j) = trace(Jk{j}*Rv);
            end
            maximize( 2*real(conj(q)*trace(J0*Rv)) ...
                     - (abs(q)^2)*( sum_square_abs(tk) + epsR ) ...
                     + rho*real(trace((u*u')*Rv)) - rho*trace(Rv) )
            subject to
                trace(Rv) == M;
     cvx_end
     R = Rv;
   rho = rho*1.5;
   if rho > 5 * 10^9
       rho = 1e9;
   end
end
[U,Sv] = eig((R+R')/2); [~,ix] = max(real(diag(Sv)));
z_hat1 = U(:,ix)/norm(U(:,ix));     

%% -------------------- MM-ISL --------------------
z_old = z;
LambdaMPhi = M - 1;
for nn = 1 : 10000
    Psi = zeros(M,M);
    for mm = 1 : numel(S)
        Psi = Psi + z_old' * JkS{mm} * z_old * (Jk{mm})';
    end
    Psi = Psi - LambdaMPhi * z_old * z_old';

    [~,D] = eig(Psi);
    LambdaMPsi = D(end,end);
    PsiNew = LambdaMPsi * eye(M) - Psi;

    PsiNew2 = PsiNew + PsiNew';

    w = PsiNew2 * z_old;
    
    z_new = sqrt(M) * (w / norm(w));
    z_old = z_new;
end
z_hat2 = z_old/norm(z_old);
%% -------------------- CAN --------------------
z_old = z;
Y = ones(2*M,1);
 for t = 1:10000
     % step 2
    znew = [z_old; zeros(M, 1)]; % 2N-by-1
    f = 1/sqrt(2*M) * fft(znew); % 2N-by-1
    v = sqrt(1/2) * exp(1i * angle(f)); % 2N-by-1
    % step 1
    g = sqrt(2*M) * ifft(v); % 2N-by-1    
    z_old = exp(1i * angle(g(1:M))); % N-by-1
 end
z_hat3 = z_old/norm(z_old);
%% -------------------- Comparisons --------------------
[ryy1, ~] = xcorr(z_hat1);
[ryy2, ~] = xcorr(z_hat2);
[ryy3, ~] = xcorr(z_hat3);
z_hat4 = z/norm(z);
[ryy4, ~] = xcorr(z_hat4);
Xlb = 1-M:M-1;
Ylb = ryy4;
Ylb2 = [];
Xlb2 = [];
for nn = 1 : numel(Xlb)
    if abs(Ylb(nn))<1e-9
    else
        Xlb2 = [Xlb2,Xlb(nn)];
        Ylb2 = [Ylb2,Ylb(nn)];
    end    
end
Nfull = numel(ryy3);
marker_idx4 = 1:3:Nfull;
marker_idx1 = [1:3:numel(Ylb2) numel(Ylb2) 29];

if marker_idx4(end) ~= Nfull
    marker_idx4 = [marker_idx4 Nfull]; % 保证最后一个点有
end
marker_idx4 = [marker_idx4, 32];
plot(Xlb2,db(abs(Ylb2)),'mx-','linewidth',1.5,'MarkerIndices', marker_idx1);hold on;
plot(1-M:M-1,db(abs(ryy3)),'>-','linewidth',1.5,'color',[0 0.5 0],'MarkerIndices', marker_idx4);
plot(1-M:M-1,db(abs(ryy2)),'bs-','linewidth',1.5, 'MarkerIndices', marker_idx4);
plot(1-M:M-1,db(abs(ryy1)),'ro-','linewidth',1.5, 'MarkerIndices', marker_idx4); 

%% Other Functions
function s = sum_cell_sqabs(Jk, R)
    s = 0; 
    for ii=1:numel(Jk)
        s = s + abs(trace(Jk{ii}*R))^2;
    end
end