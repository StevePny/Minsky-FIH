import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import rc


def f(t,x,self):
  # The equations that define the minsky ODE

  # get state vector
  lamda,omega,d_r = x
    
  # state-dependent parameters
  g_r     = self.growth_rate(omega,d_r)
  w_delta = self.wage_change(lamda)
  pi_s    = self.profit_share(omega,d_r)
  i_G     = self.investment_rate(omega,d_r)

  # ODE
  dlamda = lamda * ( g_r - (self.alpha + self.beta))
  domega = omega * (w_delta - self.alpha)
  dd_r = i_G - pi_s - d_r * g_r    

  return dlamda,domega,dd_r

def Jx(t,x,self):
  # The state-dependent Jacobian of the dynamics

  # get state vector
  lamda,omega,d_r = x
  
  df = np.zeros((len(x),len(x)))

  df[0,0] =  self.pi_S * ((1-omega-self.r*d_r)/nu - self.pi_Z)/self.nu - self.delta_K - (self.alpha + self.beta)
  df[0,1] =  -self.pi_S/nu**2
  df[0,2] =  -self.pi_S*self.r/self.nu**2 

  df[1,0] =  omega*self.lamda_S
  df[1,1] =  self.lamda_S*(lamda-self.lamda_Z) - self.alpha
  df[1,2] =  0

  df[2,0] =  0
  df[2,1] =  -self.pi_S/self.nu + 1 + d_r*self.pi_S/self.nu**2
  df[2,2] =  -self.pi_S*self.r/self.nu + self.r - (self.pi_s*((1-omega-self.r*d_r)/self.nu - self.delta_K)/self.nu - self.delta_K)
   

def Je(t,x,self):

  # Jacobian centered at the system equilibria (Appendix B, eqn. 0.20)

  # get state vector
  lamda,omega,d_r = x

  df = np.zeros((len(x),len(x)))

  ab = alpha+beta
  abd = alpha+beta+delta_K

  df[0,:] = [0, self.lamda_S*(1-(nu/ab)*((ab-self.r)*(abd/self.pi_S)*self.nu + self.pi_Z)+self.r*abd), 0]
  df[1,:] = [        -self.pi_S*(self.alpha + self.lamda_S * self.lamda_Z)/(self.nu**2 * self.lamda_S), 0 , \
                                 -(self.nu*self.delta_K + self.pi_S*(self.pi_Z-self.delta_K))/(self.nu*ab)]
  df[2,:] = [ -self.r*self.pi_S*(self.alpha + self.lamda_S * self.lamda_Z)/(self.nu**2 * self.lamda_S), 0 , \
                                 -(1/ab)*(ab**2 + self.r*(self.delta_K + self.pi_S*(self.pi_Z-self.delta_K)/self.nu))]

  return df

class minsky():

  #----------------------------------------------
  # Define default parameters (from Table 1, p.4)
  #----------------------------------------------
  def __init__(self,method='RK45',param_set='default',dt=0.01, t0=0, tf=100, \
                    alpha=0.015,beta=0.02,nu=3.0,delta_K=0.06,w_r=0.8,lamda_S=10.0,lamda_Z=0.6,pi_S=8.0,pi_Z=0.03,D=0,r=0.03):

    # Specified parameters
    self.alpha   = alpha
    self.beta    = beta
    self.nu      = nu
    self.delta_K = delta_K
    self.w_r     = w_r
    self.lamda_S = lamda_S
    self.lamda_Z = lamda_Z
    self.pi_S    = pi_S
    self.pi_Z    = pi_Z
    self.D       = D
    self.r       = r

    # Initial conditions
    self.x0 = [0.62, 0.9, 0]
    
    # Integration parameters
    self.t0 = t0
    self.tf = tf
    self.dt = dt
    self.method  = method 
    self.y       = [0,0,0]
    self.t       = 0

    # Update default parameters
    self.set_params(pset=param_set) 

    print(param_set)
    print(self)

  def __str__(self):
    print('alpha   = {}'.format(self.alpha))
    print('beta    = {}'.format(self.beta))
    print('nu      = {}'.format(self.nu))
    print('delta_K = {}'.format(self.delta_K))
    print('w_r     = {}'.format(self.w_r))
    print('lamda_S = {}'.format(self.lamda_S))
    print('lamda_Z = {}'.format(self.lamda_Z))
    print('pi_S    = {}'.format(self.pi_S))
    print('pi_Z    = {}'.format(self.pi_Z))
    print('D       = {}'.format(self.D))
    print('r       = {}'.format(self.r))
    print('method  = {}'.format(self.method))
    print('x0      = {}'.format(self.x0))
    print('t0 = {}, tf = {}, dt = {}'.format(self.t0,self.tf,self.dt))
    return ''
    

  #----------------------------------------------
  # Set parameters to match figure 1 or figure 2
  #----------------------------------------------
  def set_params(self,pset='default'):

    if (pset is 'default'):
      pass
    elif(pset is 'figure1'):
      self.tf = 400
      self.dt = 0.1
      self.x0 = [0.610, 0.850, 0.00]
      self.alpha = 0.02
      self.beta = 0.015
      self.lamda_S = 4.0
      self.pi_S =  4.6 #5.0 #SGP: 5.0 doesn't produce the expected results (contraction to the fixed point)
      self.r = 0.04
    elif(pset is 'figure2'):
      self.tf = 100
      self.dt = 0.01
      self.x0 = [0.610, 0.850, 0.00]
      self.alpha = 0.02
      self.beta = 0.015
      self.lamda_S = 4.0
      self.pi_S = 10.0
      self.r = 0.04
    else:
      print('set_params:: else')
      raise('set_params:: parameter set {} not recognized.'.format(pset))


  def growth_rate(self,omega,d_r):
    return self.pi_S*((1-omega-self.r*d_r)/self.nu - self.pi_Z)/self.nu - self.delta_K

  def profit_share(self,omega,d_r):
    return 1-omega-self.r*d_r

  def wage_change(self,lamda):
    return self.lamda_S*(lamda-self.lamda_Z)

  def investment_rate(self,omega,d_r):
    pi_s    = self.profit_share(omega,d_r)
    pi_r    = pi_s / self.nu
    return self.pi_S*(pi_r-self.pi_Z)

  def bankers_share(self,d_r):
    return d_r * self.r

  def run(self):

    # Integrate the minsky model.
    # Find additional information on integration options here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

    # lambda notation needed for older scipy versions (before 1.4.x)
    ts = np.arange(self.t0,self.tf,self.dt)
    print('run:: t0 = {}, tf = {}, dt = {}'.format(self.t0,self.tf,self.dt))
    sol = solve_ivp(lambda t, x: f(t,x,self), [self.t0,self.tf], self.x0, method=self.method, t_eval=ts)

    self.y = sol.y
    self.t = sol.t

  def plot_figure(self):

    plt.rcParams['figure.figsize'] = [10, 10]

    # Figure 1 plots:
    fig, axs = plt.subplots(3,3)

    # Remove empty plots
    axs[0,0].set_axis_off()
    axs[0,1].set_axis_off()
    axs[1,0].set_axis_off()

    # lamda, omega, and d_r over time
    axs[0,2].set_title(r'$\lambda$, $\omega$, and $d_r$ over time')
    axs[0,2].plot(self.t, self.y[0], 'k-', label = "lamda")
    axs[0,2].plot(self.t, self.y[1], 'r-', label = "omega")
    axs[0,2].plot(self.t, self.y[2], 'c-', label = "d_r")
    axs[0,2].set_xlabel('Years')
    axs[0,2].set(frame_on=False)

    # lamda vs. omega
    axs[1,1].set_title(r'$\lambda$ vs $\omega$')
    axs[1,1].plot(self.y[0], self.y[1], 'r-')
    axs[1,1].set_xlabel('Wages Share')
    axs[1,1].set_ylabel('Employment Rate')
    axs[1,1].set(frame_on=False)

    # Debt vs. wages
    axs[1,2].set_title('Debt vs Wages')
    axs[1,2].plot(self.y[1], self.y[2], 'g-')
    axs[1,2].set_xlabel('Wages Share')
    axs[1,2].set_ylabel('Debt Ratio')
    axs[1,2].set(frame_on=False)

    # Growth rate (g_r over time)
    axs[2,1].set_title('Growth Rate')
    g_r = self.growth_rate(self.y[1],self.y[2])
    axs[2,1].plot(self.t, 100*g_r, 'r-')
    axs[2,1].set_xlabel('Years')
    axs[2,1].set_ylabel('%/Year')
    axs[2,1].set(frame_on=False)

    # Debt vs. profit
    axs[2,2].set_title('Debt vs Profit')
    pi_s = self.profit_share(self.y[1],self.y[2])
    axs[2,2].plot(pi_s, self.y[2], 'b-')
    axs[2,2].set_xlabel('Profit Share')
    axs[2,2].set_ylabel('Debt Ratio')
    axs[2,2].set(frame_on=False)

    # Income distribution
    axs[2,0].set_title('Income Distribution')
    b_S = self.bankers_share(self.y[2])
    axs[2,0].plot(self.t, self.y[1], 'k-', label = "omega")
    axs[2,0].plot(self.t, b_S, 'm-', label = "b_S")
    axs[2,0].plot(self.t, pi_s, 'r-', label = "pi_S")
    axs[2,0].set_xlabel('Years')
    axs[2,0].set_ylabel('Fraction of GDP')
    axs[2,0].set(frame_on=False)

    fig.tight_layout(pad=1.0)
    plt.show()
