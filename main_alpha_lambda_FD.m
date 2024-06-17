clear all
close all

tic

N = 25; % Number of agents
M = 10; % Number of actions
T = 5000; % Time steps
q_vec = logspace(-2,0,25);
lambda = 1;
Niter = 1000;

total_wealth = zeros(Niter,length(q_vec));
gini = zeros(Niter,length(q_vec));
corr_phi_max = zeros(Niter,length(q_vec));
corr_phi_avg = zeros(Niter,length(q_vec));
corr_phi_pi = zeros(Niter,length(q_vec));
diversity = zeros(Niter,length(q_vec));
utility = zeros(Niter,length(q_vec));
corr_payoff_alpha = zeros(Niter,length(q_vec));
total_benefit = zeros(Niter,length(q_vec));
ipr_actions = zeros(Niter,length(q_vec));

p_win = [];

wealth_tmp = [];

for qq = 1:length(q_vec)
    
    q = q_vec(qq);
    
    for ni = 1:Niter
        
        p = rand(M,1); % Societal payoffs
        p_aux = cumsum(p/sum(p));

        alpha = rand(N,M); % Relative individual payoffs of actions

        %%% Assigning initial actions with probability proportional to
        %%% societal payoff
        actions = zeros(N,1);
        
        for i = 1:N
           
            aux = rand;
            
            f = find(p_aux > aux);
            f = f(1);
            
            actions(i) = f;
            
        end
        
        binc = [1:M];
        counts = hist(actions,binc);
        
        actions_freq = counts'/N;

        %%% Initial wealth

        wealth = []; 

        for i = 1:N
            wealth = [wealth; alpha(i,actions(i,end))*actions_freq(actions(i,end))];
        end
        
        %%% Initial benefit to society

        benefit = []; 
        
        for i = 1:N
            benefit = [benefit; alpha(i,actions(i,end))*p(actions(i,end))];
        end        
        
        %%% Initial probability of switching to actions
        
        ind = find(actions_freq > 0);
        c = cumsum(actions_freq(ind));        

        for t = 1:T-1

            actions_new = actions(:,end);
            wealth_new = wealth(:,end);
            benefit_new = benefit(:,end);
            
            if t > 1 
                aux_freq = actions_freq(:,end);
            else
                aux_freq = actions_freq;
            end            
            
            tmp = rand(N,1);

            for i = 1:N                

                % If random number is lower than individual payoff keep same action
                if tmp(i) < alpha(i,actions(i,end))*aux_freq(actions(i,end))

                    actions_new(i) = actions(i,end);
                    wealth_new(i) = wealth_new(i) + alpha(i,actions(i,end))*aux_freq(actions(i,end));
                    benefit_new(i) = benefit_new(i) + alpha(i,actions(i,end))*p(actions(i,end));

                    if rand < lambda
                        alpha(i,actions(i,end)) = alpha(i,actions(i,end)) + rand*(1-alpha(i,actions(i,end)));
                    end

                else % If random number is higher than individual payoff change action

                    if rand > q % Choose new action with probability proportional to crowding

                        aux = rand;
                        
                        f = find(c > aux);
                        
                        actions_new(i) = ind(f(1));
                        
                        wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*aux_freq(actions_new(i));                        
                        benefit_new(i) = benefit_new(i) + alpha(i,actions(i,end))*p(actions(i,end));

                    else % Choose new action with probability proportional to alpha

                        aux = rand;
                        
                        f = find(cumsum(alpha(i,:)/sum(alpha(i,:))) > aux);

                        actions_new(i) = f(1); 

                        wealth_new(i) = wealth_new(i) + alpha(i,actions_new(i))*aux_freq(actions_new(i));
                        benefit_new(i) = benefit_new(i) + alpha(i,actions(i,end))*p(actions(i,end));

                    end

                end

            end

            actions = [actions actions_new];
            wealth = [wealth wealth_new];
            benefit = [benefit benefit_new];
                        
            binc = [1:M];
            counts = hist(actions(:,end),binc);
        
            actions_freq = [actions_freq counts'/N];
            
            ind = find(actions_freq(:,end) > 0);
            c = cumsum(actions_freq(ind,end));            

        end
                
        total_wealth(ni,qq) = sum(wealth(:,end));
       
        total_benefit(ni,qq) = sum(benefit(:,end));
        
        aux = sort(wealth(:,end));
       
        min_wealth(ni,qq) = sum(aux(1:20));
        max_wealth(ni,qq) = sum(aux(end-20:end));
        
        diversity(ni,qq) = length(unique(actions(:,end)))/M;

        %%% Gini coefficient

        G = 0;

        for i = 1:N
            for j = i+1:N

                G = G + abs(wealth(i,end) - wealth(j,end));

            end
        end

        gini(ni,qq) = G/(N*sum(wealth(:,end)));

        %%% Inverse participation ratio

        aux_freq = aux_freq/norm(aux_freq);
        ipr_actions(ni,qq) = sum(aux_freq.^4)^-1/M;

        %%% Correlation between fitness and utility

        %%% Computing agents' fitness

        phi_avg = []; phi_max = []; phi_pi = [];

        for i = 1:N
            phi_avg = [phi_pi; mean(alpha(i,:))];
            aux = alpha(i,:).*p';
            phi_pi = [phi_pi; sum(aux)/M];
            ind = find(aux == max(aux));
            phi_max = [phi_max; aux(ind)];
        end        

        for i = 1:N
            corr_phi_avg(ni,qq) = corr(phi_avg,wealth(:,end),'type','Kendall');
            corr_phi_pi(ni,qq) = corr(phi_pi,wealth(:,end),'type','Kendall');
            corr_phi_max(ni,qq) = corr(phi_max,wealth(:,end),'type','Kendall');
        end
        
        %%% Total utility
        
        for i = 1:N
            alpha_aux = alpha(i,actions(:,end));
        end
        
        utility(ni,qq) = sum(alpha_aux.*p(actions(:,end))');
        
        corr_payoff_alpha(ni,qq) = corr(alpha_aux',p(actions(:,end)));
        
        fprintf('q = %3.2f, ni = %d\n',q,ni)
    
        f = find(actions_freq(:,end) == max(actions_freq(:,end)));
        p_win = [p_win; p(f)];
        
    end

end

toc
