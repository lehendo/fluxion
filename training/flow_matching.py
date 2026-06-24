"""
Conditional Flow Matching (CFM) loss and interpolation utilities.

Given source z_0 and target z_1 (consecutive ATLAS frames), build the
optimal-transport interpolant z_t = (1-t)*z_0 + t*z_1 and regress the
constant velocity field u_t = z_1 - z_0.
"""
# TODO: implement CFM loss, interpolation, and OT coupling
