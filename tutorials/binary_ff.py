import ngEHTforecast.fisher as fp

ff1 = fp.FF_symmetric_gaussian()
ff2 = fp.FF_asymmetric_gaussian()

print("Primary parameters:",ff1.parameter_labels())
print("Secondary parameters:",ff2.parameter_labels())

ff = fp.FF_sum([ff1,ff2])

print("Binary parameters:",ff.parameter_labels())

ffg = fp.FF_complex_gains(ff)

ffg.set_gain_epochs(scans=True)
ffg.set_gain_amplitude_prior(0.1)

print("Binary w/ gains parameters:",ffg.parameter_labels())
