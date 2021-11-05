Theory
=====================

Frequency Domain Theory
-----------------------

The overall process of RAFT is to take the design yaml input that describes the design of a FOWT and uses the input data
to fill out the 6x6 matrices of the Equations of Motion

.. math::
   X(\omega) = [-\omega^2A(\omega) + i \omega B(\omega) + C]^{-1} F(\omega)

   A(\omega) = M + A_{BEM}(\omega) + A_{morison}(\omega) + A_{aero}(\omega)
   
   B(\omega) = B_{BEM}(\omega) + B_{aero}(\omega) + B_{nonlinear-hydro-drag}(X)

   C = C_{struc} + C_{hydro} + C_{moor}

   F = F_{BEM}(\omega) + F_{hydro}(\omega) + F_{aero}(\omega) + F_{nonlinear-hydro-drag}(X)

Notice that the nonlinear-hydro-drag damping and forcing terms are a function of the platform positions, so these are iterated
until the positions (X) converge.


Aspects that RAFT does not support

- Structural flexibility (assume all members as rigid)


Member Theory
-------------

Hydrostatics
^^^^^^^^^^^^

Rectangular Moments of Inertia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Moment of Inertia about the bottom centerline point of a rectangular frustum where the upper and lower 

.. math::
   
   I_{xx} = \frac{1}{12} \rho \Big[ (W_B-W_A)^3H(\frac{L_B}{5} + \frac{L_A}{20}) + (W_B-W_A)^2 W_A H(\frac{3L_B}{4} + \frac{L_A}{4}) + \]
   \[ (W_B-W_A)W_A^2H(L_B + \frac{L_A}{2}) + \frac{1}{2}H(L_B+L_A)W_A^3 \Big] + \]
   \[ \rho \Big[ \frac{1}{5}W_BL_B + \frac{1}{20}W_AL_B + \frac{1}{20}L_AW_B + \frac{8}{15}W_AL_A \Big]

.. math::

   I_{yy} = \frac{1}{12} \rho \Big[ (L_B-L_A)^3H(\frac{W_B}{5} + \frac{W_A}{20}) + (L_B-L_A)^2 L_A H (\frac{3W_B}{4} + \frac{W_A}{4}) + \]
   \[ (L_B-L_A)L_A^2H(W_B + \frac{W_A}{2}) + \frac{1}{2}H(W_B+W_A)L_A^3 \Big] + \]
   \[ \rho \Big[ \frac{1}{5}W_BL_B + \frac{1}{20}W_AL_B + \frac{1}{20}L_AW_B + \frac{8}{15}W_AL_A \Big]

.. math::

   I_{zz} = \frac{1}{12} \rho \Big[ (L_B-L_A)^3H(\frac{W_B}{5} + \frac{W_A}{20}) + (L_B-L_A)^2 L_A H (\frac{3W_B}{4} + \frac{W_A}{4}) + \]
   \[ (L_B-L_A)L_A^2H(W_B + \frac{W_A}{2}) + \frac{1}{2}H(W_B+W_A)L_A^3 \Big] + \]
   \[ \frac{1}{12} \rho \Big[ (W_B-W_A)^3H(\frac{L_B}{5} + \frac{L_A}{20}) + (W_B-W_A)^2 W_A H(\frac{3L_B}{4} + \frac{L_A}{4}) + \]
   \[ (W_B-W_A)W_A^2H(L_B + \frac{L_A}{2}) + \frac{1}{2}H(L_B+L_A)W_A^3 \Big]






The theory behind RAFT is in the process of being written up and published. 
Please check back later or contact us if in need of a specific clarification.
