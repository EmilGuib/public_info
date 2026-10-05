import sympy as sp

def Exact_Angular_Flux_Pure_Absorber(angular_flux_in,sigma_T,s,mu):
    return angular_flux_in*sp.exp(-sigma_T*s/mu)

def Exact_Angular_Flux_Uniform_Source(angular_flux_eq,sigma_T,s,mu):
    return angular_flux_eq*(1-sp.exp(-sigma_T*s/mu))

def Average_Cell_True_Angular_Flux(angular_flux,cell_width,s):
    return (1/cell_width)*sp.integrate(angular_flux,(s,0,cell_width))

def Weighted_Difference(angular_flux_in,angular_flux_eq,tau,a):
    angular_flux_out = (tau*angular_flux_eq-angular_flux_in*(tau*(1-a)-1))/(1+tau*a)
    cell_average_angular_flux = (1-a)*angular_flux_in+a*angular_flux_out
    return angular_flux_out,cell_average_angular_flux

def Discontinuous_Galerkin(angular_flux_in,angular_flux_eq,cell_width,mu,sigma_T):
    q = angular_flux_eq*sigma_T
    c = sp.Matrix([[mu+sigma_T*cell_width,mu],[-mu,mu+sigma_T*cell_width/3]])
    b = sp.Matrix([q*cell_width+mu*angular_flux_in,-mu*angular_flux_in])
    return c.LUsolve(b)

########################################################################
#symbols for sympy library
psi_in, psi_eq= sp.symbols('psi_in psi_eq')
tau, sigma_T, s, mu, cell_width = sp.symbols('tau sigma_t s mu h', positive=True)
########################################################################
#exact angular fluxes for comparison
exact_angular_flux_absorber = Exact_Angular_Flux_Pure_Absorber(psi_in,sigma_T,s,mu)
exact_angular_flux_out_absorber = Exact_Angular_Flux_Pure_Absorber(psi_in,sigma_T,cell_width,mu)
exact_average_angular_flux_absorber = Average_Cell_True_Angular_Flux(exact_angular_flux_absorber,cell_width,s)

exact_angular_flux_out_absorber = sp.simplify(exact_angular_flux_out_absorber.subs({cell_width: tau*mu/sigma_T}))
exact_average_angular_flux_absorber = sp.simplify(exact_average_angular_flux_absorber.subs({cell_width: tau*mu/sigma_T}))


exact_angular_flux_source = Exact_Angular_Flux_Uniform_Source(psi_eq,sigma_T,s,mu)
exact_angular_flux_out_source = Exact_Angular_Flux_Uniform_Source(psi_eq,sigma_T,cell_width,mu)
exact_average_angular_flux_source = Average_Cell_True_Angular_Flux(exact_angular_flux_source,cell_width,s)

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
#Test #1: Pure Absorber with Incident Flux

DD_angular_flux_out, DD_average_angular_flux = Weighted_Difference(psi_in,0,tau,sp.Rational(1,2))
SD_angular_flux_out, SD_average_angular_flux = Weighted_Difference(psi_in,0,tau,sp.Rational(1,1))


DD_angular_flux_out=sp.simplify(DD_angular_flux_out)
DD_average_angular_flux=sp.simplify(DD_average_angular_flux)
SD_angular_flux_out=sp.simplify(SD_angular_flux_out)
SD_average_angular_flux=sp.simplify(SD_average_angular_flux)

print("\n"*4)
print("DD Angular Flux Out: Experiment 1")
sp.pprint(DD_angular_flux_out.series(tau,0,5))
print("DD Average Angular Flux : Experiment 1")
sp.pprint(DD_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term DD Angular Flux Out:",(DD_angular_flux_out - exact_angular_flux_out_absorber).series(tau,0,5))
print("Error Term DD Average Angular Flux:",(DD_average_angular_flux - exact_average_angular_flux_absorber).series(tau,0,5))


print("\n"*4)
print("SD Angular Flux Out: Experiment 1")
sp.pprint(SD_angular_flux_out.series(tau,0,5))
print("SD Average Angular Flux : Experiment 1")
sp.pprint(SD_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term SD Angular Flux Out:",(SD_angular_flux_out - exact_angular_flux_out_absorber).series(tau,0,5))
print("Error Term SD Average Angular Flux:",(SD_average_angular_flux - exact_average_angular_flux_absorber).series(tau,0,5))


####################################################################
#Test #2: Vacuum Boundary With Source

DD_angular_flux_out, DD_average_angular_flux = Weighted_Difference(0,psi_eq,tau,sp.Rational(1,2))
SD_angular_flux_out, SD_average_angular_flux = Weighted_Difference(0,psi_eq,tau,sp.Rational(1,1))


DD_angular_flux_out=sp.simplify(DD_angular_flux_out)
DD_average_angular_flux=sp.simplify(DD_average_angular_flux)
SD_angular_flux_out=sp.simplify(SD_angular_flux_out)
SD_average_angular_flux=sp.simplify(SD_average_angular_flux)

print("\n"*4)
print("DD Angular Flux Out: Experiment 2")
sp.pprint(DD_angular_flux_out.series(tau,0,5))
print("DD Average Angular Flux : Experiment 2")
sp.pprint(DD_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term DD Angular Flux Out:",(DD_angular_flux_out - exact_angular_flux_out_source).series(tau,0,5))
print("Error Term DD Average Angular Flux:",(DD_average_angular_flux - exact_average_angular_flux_source).series(tau,0,5))


print("\n"*4)
print("SD Angular Flux Out: Experiment 2")
sp.pprint(SD_angular_flux_out.series(tau,0,5))
print("SD Average Angular Flux : Experiment 2")
sp.pprint(SD_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term SD Angular Flux Out:",(SD_angular_flux_out - exact_angular_flux_out_source).series(tau,0,5))
print("Error Term SD Average Angular Flux:",(SD_average_angular_flux - exact_average_angular_flux_source).series(tau,0,5))

####################################################################
####################################################################
#Discontinuous Galerkin Method 

#Test 1: Pure Absorber
DG_angular_flux_components = Discontinuous_Galerkin(psi_in,0,cell_width,mu,sigma_T)

DG_average_angular_flux = DG_angular_flux_components[0]
DG_angular_flux_out = DG_angular_flux_components[0]+DG_angular_flux_components[1]

DG_average_angular_flux=sp.simplify(DG_average_angular_flux.subs({cell_width: tau*mu/sigma_T}))
DG_angular_flux_out=sp.simplify(DG_angular_flux_out.subs({cell_width: tau*mu/sigma_T}))

print("\n"*4)
print("DG Angular Flux Out: Experiment 1")
sp.pprint(DG_angular_flux_out.series(tau,0,5))
print("DG Average Angular Flux: Experiment 1")
sp.pprint(DG_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term DG Angular Flux Out:",(DG_angular_flux_out - exact_angular_flux_out_absorber).series(tau,0,5))
print("Error Term DG Average Angular Flux:",(DG_average_angular_flux - exact_average_angular_flux_absorber).series(tau,0,5))


####################################################################
#Test 2: Vaccum Inflow:
DG_angular_flux_components = Discontinuous_Galerkin(0,psi_eq,cell_width,mu,sigma_T)

DG_average_angular_flux = DG_angular_flux_components[0]
DG_angular_flux_out = DG_angular_flux_components[0]+DG_angular_flux_components[1]

DG_average_angular_flux=sp.simplify(DG_average_angular_flux.subs({cell_width: tau*mu/sigma_T}))
DG_angular_flux_out=sp.simplify(DG_angular_flux_out.subs({cell_width: tau*mu/sigma_T}))

print("\n"*4)
print("DG Angular Flux Out: Experiment 2")
sp.pprint(DG_angular_flux_out.series(tau,0,5))
print("DG Average Angular Flux: Experiment 2")
sp.pprint(DG_average_angular_flux.series(tau,0,5))
print("\n")
print("Error Term DG Angular Flux Out:",(DG_angular_flux_out - exact_angular_flux_out_source).series(tau,0,5))
print("Error Term DG Average Angular Flux:",(DG_average_angular_flux - exact_average_angular_flux_source).series(tau,0,5))
