from rsatb import LCAOIntegralInput, LCAOModel, PreconditionerOptions, RSATB, SetupOptions

kpoints = [
    [0. , 0. , 0. ]
]

model_runner = RSATB.from_file(npar=1, filename="./data/diamond/diamond_sc111.vasp", r_max_per_type=[3.7], type_norb=[13], verbose=True)

model_runner.setup_lcao(
    LCAOModel.hdf5(
        "./data/diamond/C2.0/hamiltonians.h5",
        integrals=LCAOIntegralInput(
            orb_names=["C_gga_7au_100Ry_2s2p1d.orb"],
            orb_dir="./data/diamond/",
            psd_names=["C_ONCV_PBE-1.0.upf"],
            psd_dir="./data/diamond/",
        ),
    ),
    SetupOptions(
        kpoints=kpoints,
        nsamples=1,
        preconditioner=PreconditionerOptions(
            mode="S",
            tol=1e-7,
            sptol=1e-5,
            maxiter=10,
            method="graph",
        ),
    ),
)

model_runner.test_solve()
