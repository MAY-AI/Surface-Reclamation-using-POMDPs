## Surface Reclamation using POMDPs
# Gabriel Agostine
# University of Colorado Boulder

# Date: 5/5/2025

# Descritpion: Humanity is on the brink of extinction and has been 
#              forced to survive underground after the relentless 
#              assault of an unknown force that has descended on humanity. 
#              This code describes a Partially Observable Markov 
#              Decision Process (POMDP) formulation for a resource 
#              management reinforcement learning agent whose goal is to 
#              reclaim the surface from said unknown force. Operating 
#              with a handful of squads, the agent must learn to manage 
#              limited resources, deploy teams to explore different regions, 
#              make strategic decisions under uncertainty, and maintain the 
#              sanity and comfort of its people.

# GitHub: https://github.com/MAY-AI/Surface-Reclamation-using-POMDPs/tree/main

## Imports
##################################################################################
##################################################################################

using POMDPs
using QuickPOMDPs
using POMDPTools
using Distributions
using ParticleFilters
using Random
using BasicPOMCP
using QMDP
using Plots

## CONSTANTS
##################################################################################
##################################################################################

N = 1
D = 2
BASE_SUCCESS_RATES = [0.8, 0.8, 0.7, 0.6, 0.5]
HEALTH_MODIFIERS = [0.6, 0.9, 1.1]
RESOURCE_YIELDS = [
    ((20, 20), (30, 30), (20, 20), (40, 40)),  # Region 1
    ((30, 30), (40, 40), (30, 30), (50, 50)),  # Region 2
    ((40, 40), (50, 50), (40, 40), (60, 60)),  # Region 3
    ((50, 50), (60, 60), (50, 50), (70, 70)),  # Region 4
    ((60, 60), (70, 70), (60, 60), (80, 80))   # Region 5+
]
RESOURCE_DECAY = [5, 5, 5, 5]
RESOURCE_MINIMUMS = [60, 60, 60, 60]
ALPHA = 0.5
ALPHA_r = 25
BETA_r = 5  
GAMMA_r = 10
DELTA_r = 5
EPSILON_r = 10
ZETA_r = 10
last_region = zeros(Int, N)

## FUNCTIONS
##################################################################################
##################################################################################

function clamp_resources(r::Float64)
    floor(Int, (r - 1e-6) / 5) * 5
end
function clamp_resources(r::Int)
    ((r - 1) ÷ 5) * 5
end
function clamp_resources(r::Vector{Float64})
    floor.(Int, (r .- 1e-6) ./ 5) .* 5
end
function clamp_resources(r::Vector{Int})
    ((r .- 1) .÷ 5) .* 5
end
function AnalyzeHistory(hist)
    food = [s[1] for s in state_hist(hist)]
    med = [s[2] for s in state_hist(hist)]
    fuel = [s[3] for s in state_hist(hist)]
    materials = [s[4] for s in state_hist(hist)]
    squad_health = [s[4+D+N+1:4+D+2N] for s in state_hist(hist)]
    tech_levels = [s[end] for s in state_hist(hist)]
    rewards = collect(reward_hist(hist))
    
    p1 = plot(title="Resource Values", xlabel="Step", ylabel="Resource Stockpiles")
    # Initialize plot
    
    plot!(p1, food, label="Food", linewidth=2)
    plot!(p1, med, label="Medicine", linewidth=2)
    plot!(p1, fuel, label="Fuel", linewidth=2)
    plot!(p1, materials, label="Materials", linewidth=2)
    # Create resource plots
    
    display(p1)
    savefig("Final Project\\state_evolution.png")
    # Save figure

    p2 = plot(title="Squad Health Over Time", xlabel="Step", ylabel="Health")
    # Plot squad health

    for i in 1:N
        member_health = [sh[i] for sh in squad_health]
        plot!(p2, member_health, label="Squad $i", linewidth=2)
    end
    # Create squad plots

    display(p2)
    savefig(p2, "Final Project\\squad_health.png")
    # Save figure

    # Tech Level Plot (new)
    p3 = plot(title="Technology Progression", xlabel="Step", ylabel="Tech Level"; legend=false)
    plot!(p3, tech_levels, linewidth=2)

    display(p3)
    savefig(p3, "Final Project\\tech_level.png")
    # Save figure

    # Reward Plot (new)
    p4 = plot(title="Reward Over Time", xlabel="Step", ylabel="Immediate Reward"; legend=false)
    plot!(p4, rewards, linewidth=2)

    display(p4)
    savefig(p4, "Final Project\\reward.png")
    # Save figure

end

## POMDP Formulation (S, A, O, R, T, Z, gamma)
##################################################################################
##################################################################################

surface_reclamation = QuickPOMDP(
    states = [
        (r_food, r_med, r_fuel, r_mat, t..., q..., h..., τ)
        for (r_food, r_med, r_fuel, r_mat, t, q, h, τ) in Iterators.product(
            0:5:100, 0:5:100, 0:5:100, 0:5:100,
            Iterators.product(fill(0:1, D)...),
            Iterators.product(fill(0:1, N)...),
            Iterators.product(fill(0:2, N)...),
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
    ],
    
    observations = [
        (r_food, r_med, r_fuel, r_mat, t..., q..., h..., τ)
        for (r_food, r_med, r_fuel, r_mat, t, q, h, τ) in Iterators.product(
            0:5:100, 0:5:100, 0:5:100, 0:5:100,
            Iterators.product(fill(0:1, D)...),
            Iterators.product(fill(0:1, N)...),
            Iterators.product(fill(0:2, N)...),
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        )
    ],

    # Dynamically generated actions
    actions = vcat(
        vec([Symbol("deploy_$(i)_$(j)") for i in 1:N, j in 1:D]),
        [Symbol("recall_$(i)") for i in 1:N],
        [Symbol("heal_$(i)") for i in 1:N],
        [Symbol("support_$(i)") for i in 1:N],
        [:research_tech],
        [Symbol("research_region_$(j)") for j in 2:D]
    ),
    
    transition = function (s, a)
        # @show a
        # Parse state components
        resources = collect(s[1:4])
        territories = collect(s[5:4+D])
        squad_status = collect(s[4+D+1:4+D+N])
        squad_health = collect(s[4+D+N+1:4+D+2N])
        τ = s[end]
        
        action_parts = split(string(a), "_")
        action_type = action_parts[1]
        valid_action = true
        new_states = []
        probs = []
        # @show s
        
        # Deployment Action (Section 3.5.1)
        if action_type == "deploy"
            squad_idx = parse(Int, action_parts[2])
            region_idx = parse(Int, action_parts[3])
            region_idx = min(region_idx, length(BASE_SUCCESS_RATES))  # Handle "Else"
            
            # Check preconditions
            fuel_cost = 5 * region_idx
            if squad_status[squad_idx] != 1 || territories[region_idx] != 1 || resources[3] < fuel_cost
                valid_action = false
            else
                resources[3] = clamp_resources(resources[3] - fuel_cost)
                squad_status[squad_idx] = 0
                last_region[squad_idx] = region_idx
                
                # Calculate success probability
                base_success = BASE_SUCCESS_RATES[region_idx]
                health_mod = HEALTH_MODIFIERS[squad_health[squad_idx]+1]
                success_prob = base_success * health_mod
                partial_prob = 0.05 * base_success * health_mod
                
                # Success outcome
                success_state = collect(s)
                yields = [rand(r[1]:r[2]) for r in RESOURCE_YIELDS[region_idx]]
                success_state[1:4] = clamp_resources(min.(resources .+ yields .* (1+τ), 100))
                push!(new_states, tuple(success_state...))
                push!(probs, success_prob)

                # Partial success
                partial_state = collect(s)
                partial_state[1:4] = clamp_resources(min.(resources .+ ALPHA .* yields .* (1+τ), 100))
                if rand() < 0.4
                    partial_state[4+D+N+squad_idx] = max(squad_health[squad_idx]-1, 0)
                end
                push!(new_states, tuple(partial_state...))
                push!(probs, partial_prob)

                # Failure
                fail_state = collect(s)
                fail_state[1:4] = resources
                if rand() < 0.7
                    fail_state[4+D+N+squad_idx] = max(squad_health[squad_idx]-1, 0)
                end
                push!(new_states, tuple(fail_state...))
                push!(probs, 1 - success_prob - partial_prob)
            end

        # Recall Action (Section 3.5.2)
        elseif action_type == "recall"
            squad_idx = parse(Int, action_parts[2])
            new_state = collect(s)
            
            # Calculate recall cost based on last deployment
            fuel_cost = 5 * last_region[squad_idx]
            last_region[squad_idx] = 0
            new_state[3] = clamp_resources(max(new_state[3] - fuel_cost, 0))
            new_state[4+D+squad_idx] = 1 # Squad available
            
            # 1% damage chance during recall
            if rand() < 0.01
                new_state[4+D+N+squad_idx] = max(new_state[4+D+N+squad_idx]-1, 0)
            end
            
            push!(new_states, tuple(new_state...))
            push!(probs, 1.0)
    
        # Healing Action (Section 3.5.3)
        elseif action_type == "heal"
            squad_idx = parse(Int, action_parts[2])
            new_state = collect(s)
            current_health = new_state[4+D+N+squad_idx]
            
            med_cost = current_health == 0 ? 15 : 25
            if new_state[2] < med_cost || current_health == 2
                valid_action = false
            else
                new_state[2] = clamp_resources(max(new_state[2] - med_cost, 0))
                
                if current_health == 0
                    new_state[4+D+N+squad_idx] = 1
                elseif current_health == 1
                    new_state[4+D+N+squad_idx] = rand() < 0.85 ? 2 : 1
                end
                
                push!(new_states, tuple(new_state...))
                push!(probs, 1.0)
            end
    
        # Field Support Action (Section 3.5.4)
        elseif action_type == "support"
            squad_idx = parse(Int, action_parts[2])
            new_state = collect(s)
            
            if new_state[1] < 10 || new_state[2] < 15
                valid_action = false
            else
                new_state[1] = clamp_resources(max(new_state[1] - 10, 0))
                new_state[2] = clamp_resources(max(new_state[2] - 15, 0))
                
                if rand() < 0.85
                    if rand() < 0.6
                        new_state[4+D+N+squad_idx] = min(new_state[4+D+N+squad_idx]+1, 2)
                    end
                    if rand() < 0.15
                        new_state[end] = round(min(new_state[end] + 0.1, 0.5), digits=1)
                    end
                end
                
                push!(new_states, tuple(new_state...))
                push!(probs, 1.0)
            end

        # Research Tech Action (Section 3.5.5)
        elseif action_type == "research" && action_parts[2] == "tech"
            new_state = collect(s)
            mat_cost = 40
            if new_state[4] < mat_cost
                valid_action = false
            else
                new_state[4] = clamp_resources(max(new_state[4] - mat_cost, 0))
                
                if rand() < 0.9
                    new_state[end] = round(min(new_state[end] + 0.1, 0.5), digits=1)
                end
                
                push!(new_states, tuple(new_state...))
                push!(probs, 1.0)
            end

        # Research Region Action (Section 3.5.6)
        elseif action_type == "research" && action_parts[2] == "region"
            region_idx = parse(Int, action_parts[3])
            new_state = collect(s)
            mat_cost = 20 * region_idx
            if new_state[4] < mat_cost
                valid_action = false
            else
                new_state[4] = clamp_resources(max(new_state[4] - mat_cost, 0))
                
                progress_prob = 0.7 + τ*10
                if rand() < progress_prob
                    new_state[4+region_idx] = 1 # Unlock region
                end
                
                push!(new_states, tuple(new_state...))
                push!(probs, 1.0)
            end
    
        else
            error("Invalid action: $a")
        end

        # Environmental Transitions (Section 3.5.8)
        if valid_action && !isempty(new_states)
            for i in 1:length(new_states)
                s_new = collect(new_states[i])
                s_new[end] = round(s_new[end], digits=1)
                s_new[1:4] = clamp_resources(max.(s_new[1:4] .- RESOURCE_DECAY, 0))
                for squad in 1:N
                    if s_new[4+D+squad] == 1 && rand() < 0.2
                        s_new[4+D+N+squad] = min(s_new[4+D+N+squad]+1, 2)
                    end
                end
                new_states[i] = tuple(s_new...)
            end
        else
            return SparseCat([s], [1.0])  # Invalid action, no change
        end

        return SparseCat(new_states, probs)
    end,

    observation = function (a, sp)
        # @show a 
        # @show sp 
        obs_t = ntuple(j -> Int(rand() < 0.8 ? sp[4+j] : 1 - sp[4+j]), D)
        squad_stat = ntuple(i -> Int(sp[4+D+i]), N)
        obs_h = ntuple(i -> Int(rand() < 0.7 ? sp[4+D+N+i] : rand(0:2)), N)
        o = (
            Int(sp[1]), Int(sp[2]), Int(sp[3]), Int(sp[4]),
            obs_t..., 
            squad_stat..., 
            obs_h...,
            Float64(sp[end])
        )::obstype(surface_reclamation)
        # @show o
        return Deterministic(o)
    end,
    
    reward = function (s, a, sp)
        territories = s[5:4+D]
        resources = s[1:4]
        s_prev_health = s[4+D+N+1 : 4+D+2N]
        s_current_health = sp[4+D+N+1 : 4+D+2N]
        prev_tech = s[end]
        current_tech = sp[end]
    
        new_territories = sum(territories)
        resource_penalty = -GAMMA_r*(
            max(RESOURCE_MINIMUMS[1] - resources[1], 0) +
            max(RESOURCE_MINIMUMS[2] - resources[2], 0) +
            max(RESOURCE_MINIMUMS[3] - resources[3], 0) +
            max(RESOURCE_MINIMUMS[4] - resources[4], 0)
        )
        health_penalty = -DELTA_r * sum(max.(0, s_prev_health .- s_current_health))
        research_reward = EPSILON_r * (current_tech - prev_tech)
        
        return ALPHA_r*new_territories + BETA_r*sum(territories) + resource_penalty + health_penalty + research_reward + ZETA_r * new_territories
    end,
    
    initialstate = Deterministic((
        100, 100, 100, 100,     # Full resources
        1, zeros(Int, D-1)...,  # Only first region
        ones(Int, N)...,        # All squads available
        fill(1, N)...,          # Optimal health
        round(0.0, digits=1)    # Initial tech
    )),
    
    discount = 0.99,

    isterminal = s -> false

)

## Algorithm
##################################################################################
##################################################################################

function EvaluatePOMDP()
    println("Initializing POMCP solver...")
    pomcp_solver = POMCPSolver(
        tree_queries=20_000,
        max_depth=15,
        c=100.0,
        max_time=10.0,
        estimate_value=RolloutEstimator(RandomSolver(MersenneTwister(42))),
        rng=MersenneTwister(42)
    )

    println("Initializing particle filter...")
    up = ParticleFilters.BootstrapFilter(surface_reclamation, 5_000, rng=MersenneTwister(42))

    println("Solving...")
    policy = solve(pomcp_solver, surface_reclamation)
    
    println("Simulating...\n")
    hist = simulate(
        HistoryRecorder(max_steps=100, show_progress=true),
        surface_reclamation,
        policy,
        up
    )

    println("Discounted return: ", discounted_reward(hist))
    return hist
end

## Main
##################################################################################
##################################################################################

hist = EvaluatePOMDP()
# Evaluate POMDP

# AnalyzeHistory(hist)
# Plot results