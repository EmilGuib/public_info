import sympy as sp

def exact_angular_flux_pure_absorber(angular_flux_in,sigma_T,s,mu):
    return angular_flux_in*sp.exp(-sigma_T*s/mu)

def exact_angular_flux_uniform_source(angular_flux_eq,sigma_T,s,mu):
    return angular_flux_eq*(1-sp.exp(-sigma_T*s/mu))

def average_cell_exact_angular_flux(angular_flux,cell_width,s):
    return (1/cell_width)*sp.integrate(angular_flux,(s,0,cell_width))

def weighted_difference(angular_flux_in,angular_flux_eq,tau,a):
    angular_flux_out = (tau*angular_flux_eq-angular_flux_in*(tau*(1-a)-1))/(1+tau*a)
    cell_average_angular_flux = (1-a)*angular_flux_in+a*angular_flux_out
    return angular_flux_out,cell_average_angular_flux

def discontinuous_galerkin(angular_flux_in,angular_flux_eq,cell_width,mu,sigma_T):
    q = angular_flux_eq*sigma_T
    system_matrix = sp.Matrix([[mu+sigma_T*cell_width,mu],[-mu,mu+sigma_T*cell_width/3]])
    system_rhs = sp.Matrix([q*cell_width+mu*angular_flux_in,-mu*angular_flux_in])
    return system_matrix.LUsolve(system_rhs)

def print_series_error(scheme_out,scheme_average,exact_out,exact_average,label,experiment,tau):
    print("\n"*4)
    print(f"{label} Angular Flux Out: Experiment {experiment}")
    sp.pprint(scheme_out.series(tau,0,5))
    print(f"{label} Average Angular Flux: Experiment {experiment}")
    sp.pprint(scheme_average.series(tau,0,5))
    print(f"Leading Error Term {label} Angular Flux Out:",(scheme_out - exact_out).as_leading_term(tau))
    print(f"Leading Error Term {label} Average Angular Flux:",(scheme_average - exact_average).as_leading_term(tau))
    
def main():
    psi_in, psi_eq= sp.symbols('psi_in psi_eq')
    tau, sigma_T, s, mu, cell_width = sp.symbols('tau sigma_t s mu h', positive=True)
    ########################################################################
    #exact angular fluxes for comparison
    exact_angular_flux_absorber = exact_angular_flux_pure_absorber(psi_in,sigma_T,s,mu)
    exact_angular_flux_out_absorber = exact_angular_flux_pure_absorber(psi_in,sigma_T,cell_width,mu)
    exact_average_angular_flux_absorber = average_cell_exact_angular_flux(exact_angular_flux_absorber,cell_width,s)

    exact_angular_flux_out_absorber = sp.simplify(exact_angular_flux_out_absorber.subs({cell_width: tau*mu/sigma_T}))
    exact_average_angular_flux_absorber = sp.simplify(exact_average_angular_flux_absorber.subs({cell_width: tau*mu/sigma_T}))


    exact_angular_flux_source = exact_angular_flux_uniform_source(psi_eq,sigma_T,s,mu)
    exact_angular_flux_out_source = exact_angular_flux_uniform_source(psi_eq,sigma_T,cell_width,mu)
    exact_average_angular_flux_source = average_cell_exact_angular_flux(exact_angular_flux_source,cell_width,s)

    exact_angular_flux_out_source = sp.simplify(exact_angular_flux_out_source.subs({cell_width: tau*mu/sigma_T}))
    exact_average_angular_flux_source = sp.simplify(exact_average_angular_flux_source.subs({cell_width: tau*mu/sigma_T}))


    print("Exact Angular Flux Out: Experiment 1")
    sp.pprint(exact_angular_flux_out_absorber.series(tau,0,5))
    print("Exact Average Angular Flux: Experiment 1")
    sp.pprint(exact_average_angular_flux_absorber.series(tau,0,5))
    print("\n"*4)
    print("Exact Angular Flux Out: Experiment 2")
    sp.pprint(exact_angular_flux_out_source.series(tau,0,5))
    print("Exact Average Angular Flux: Experiment 2")
    sp.pprint(exact_average_angular_flux_source.series(tau,0,5))
    ####################################################################
    ####################################################################
    #WD and SD Methods
    
    #Test #1: Pure Absorber with Incident Flux

    DD_angular_flux_out, DD_average_angular_flux = weighted_difference(psi_in,0,tau,sp.Rational(1,2))
    SD_angular_flux_out, SD_average_angular_flux = weighted_difference(psi_in,0,tau,sp.Rational(1,1))


    DD_angular_flux_out = sp.simplify(DD_angular_flux_out)
    DD_average_angular_flux = sp.simplify(DD_average_angular_flux)
    SD_angular_flux_out = sp.simplify(SD_angular_flux_out)
    SD_average_angular_flux = sp.simplify(SD_average_angular_flux)

    print_series_error(DD_angular_flux_out,DD_average_angular_flux,
                       exact_angular_flux_out_absorber,exact_average_angular_flux_absorber,"DD","1",tau)
    
    print_series_error(SD_angular_flux_out,SD_average_angular_flux,
                           exact_angular_flux_out_absorber,exact_average_angular_flux_absorber,"SD","1",tau)

    ####################################################################
    #Test #2: Vacuum Boundary With Source

    DD_angular_flux_out, DD_average_angular_flux = weighted_difference(0,psi_eq,tau,sp.Rational(1,2))
    SD_angular_flux_out, SD_average_angular_flux = weighted_difference(0,psi_eq,tau,sp.Rational(1,1))


    DD_angular_flux_out = sp.simplify(DD_angular_flux_out)
    DD_average_angular_flux = sp.simplify(DD_average_angular_flux)
    SD_angular_flux_out = sp.simplify(SD_angular_flux_out)
    SD_average_angular_flux = sp.simplify(SD_average_angular_flux)

    print_series_error(DD_angular_flux_out,DD_average_angular_flux,
                       exact_angular_flux_out_source,exact_average_angular_flux_source,"DD","2",tau)
    
    print_series_error(SD_angular_flux_out,SD_average_angular_flux,
                           exact_angular_flux_out_source,exact_average_angular_flux_source,"SD","2",tau)
    ####################################################################
    ####################################################################
    #Discontinuous Galerkin Method 

    #Test 1: Pure Absorber
    DG_angular_flux_components = discontinuous_galerkin(psi_in,0,cell_width,mu,sigma_T)

    DG_average_angular_flux = DG_angular_flux_components[0]
    DG_angular_flux_out = DG_angular_flux_components[0]+DG_angular_flux_components[1]

    DG_average_angular_flux = sp.simplify(DG_average_angular_flux.subs({cell_width: tau*mu/sigma_T}))
    DG_angular_flux_out = sp.simplify(DG_angular_flux_out.subs({cell_width: tau*mu/sigma_T}))

    print_series_error(DG_angular_flux_out,DG_average_angular_flux,
                       exact_angular_flux_out_absorber,exact_average_angular_flux_absorber,"DG","1",tau)
    

    ####################################################################
    #Test #2: Vacuum Boundary With Source
    DG_angular_flux_components = discontinuous_galerkin(0,psi_eq,cell_width,mu,sigma_T)

    DG_average_angular_flux = DG_angular_flux_components[0]
    DG_angular_flux_out = DG_angular_flux_components[0]+DG_angular_flux_components[1]

    DG_average_angular_flux = sp.simplify(DG_average_angular_flux.subs({cell_width: tau*mu/sigma_T}))
    DG_angular_flux_out = sp.simplify(DG_angular_flux_out.subs({cell_width: tau*mu/sigma_T}))

    print_series_error(DG_angular_flux_out,DG_average_angular_flux,
                       exact_angular_flux_out_source,exact_average_angular_flux_source,"DG","2",tau)

if __name__ == "__main__":
    main()