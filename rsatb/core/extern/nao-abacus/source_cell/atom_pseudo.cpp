#include "atom_pseudo.h"
#include "soc.h"
#include "source_basis/module_nao/numerical_radial.h"
#include <algorithm>
#include "source_base/parallel_common.h"
#include <cassert>
// #include "source_io/module_parameter/parameter.h"

Atom_pseudo::Atom_pseudo()
{
}

Atom_pseudo::~Atom_pseudo()
{
}

// mohan add 2021-05-07
void Atom_pseudo::set_d_so(ModuleBase::ComplexMatrix& d_so_in,
                           const int& nproj_in,
                           const int& nproj_in_so,
                           const bool has_so,
                           const bool lspinorb, // zyzh
                           const int& nspin
                           )
{
    if (this->lmax < -1 || this->lmax > 20)
    {
        throw std::runtime_error("Atom_pseudo: bad input of lmax : should be between -1 and 20");
    }

    this->nproj = nproj_in;
    this->nproj_soc = nproj_in_so;
    int spin_dimension = 4;

    // optimize
    for (int is = 0; is < spin_dimension; is++)
    {
        this->non_zero_count_soc[is] = 0;
        this->index1_soc[is] = std::vector<int>(nproj_soc * nproj_soc, 0);
        this->index2_soc[is] = std::vector<int>(nproj_soc * nproj_soc, 0);
    }

    if (!has_so)
    {
        this->d_real.create(nproj_soc + 1, nproj_soc + 1);
        this->d_so.create(spin_dimension, nproj_soc + 1, nproj_soc + 1); // for noncollinear-spin only case

        // calculate the number of non-zero elements in dion
        for (int L1 = 0; L1 < nproj_soc; L1++)
        {
            for (int L2 = 0; L2 < nproj_soc; L2++)
            {
                this->d_real(L1, L2) = d_so_in(L1, L2).real();
                if (std::fabs(d_real(L1, L2)) > 1.0e-8)
                {
                    this->index1_soc[0][non_zero_count_soc[0]] = L1;
                    this->index2_soc[0][non_zero_count_soc[0]] = L2;
                    this->non_zero_count_soc[0]++;
                }
                // for noncollinear-spin only case
                this->d_so(0, L1, L2) = d_so_in(L1, L2);
                this->d_so(3, L1, L2) = d_so_in(L1, L2);
                if (std::fabs(d_real(L1, L2)) > 1.0e-8)
                {
                    this->index1_soc[3][non_zero_count_soc[3]] = L1;
                    this->index2_soc[3][non_zero_count_soc[3]] = L2;
                    this->non_zero_count_soc[3]++;
                }
            }
        }
    }
    else // zhengdy-soc
    {
        this->d_so.create(spin_dimension, nproj_soc + 1, nproj_soc + 1);
        //		std::cout << "lmax=" << lmax << std::endl;

        if (this->lmax > -1)
        {
            if (lspinorb)
            {
                int is = 0;
                for (int is1 = 0; is1 < 2; is1++)
                {
                    for (int is2 = 0; is2 < 2; is2++)
                    {
                        for (int L1 = 0; L1 < nproj_soc; L1++)
                        {
                            for (int L2 = 0; L2 < nproj_soc; L2++)
                            {
                                this->d_so(is, L1, L2) = d_so_in(L1 + nproj_soc * is1, L2 + nproj_soc * is2);

                                if (fabs(this->d_so(is, L1, L2).real()) > 1.0e-8
                                    || fabs(this->d_so(is, L1, L2).imag()) > 1.0e-8)
                                {
                                    //									std::cout << "tt in atom is=" << is << " L1=" <<
                                    //L1
                                    //<< " L2="
                                    //									<< L2 << " " << d_so(is, L1, L2) << std::endl;

                                    this->index1_soc[is][non_zero_count_soc[is]] = L1;
                                    this->index2_soc[is][non_zero_count_soc[is]] = L2;
                                    this->non_zero_count_soc[is]++;
                                }
                            }
                        }
                        is++;
                    }
                }
            }
            else
            {
                int is = 0;
                for (int is1 = 0; is1 < 2; is1++)
                {
                    for (int is2 = 0; is2 < 2; is2++)
                    {
                        if (is >= nspin) {
                            break;
}
                        for (int L1 = 0; L1 < nproj_soc; L1++)
                        {
                            for (int L2 = 0; L2 < nproj_soc; L2++)
                            {
                                if (is == 1 || is == 2)
                                {
                                    this->d_so(is, L1, L2) = std::complex<double>(0.0, 0.0);
                                }
                                else
                                {
                                    this->d_so(is, L1, L2) = d_so_in(L1 + nproj_soc * is1, L2 + nproj_soc * is2);
                                }
                                if (std::abs(this->d_so(is, L1, L2).real()) > 1.0e-8
                                    || std::abs(this->d_so(is, L1, L2).imag()) > 1.0e-8)
                                {
                                    this->index1_soc[is][non_zero_count_soc[is]] = L1;
                                    this->index2_soc[is][non_zero_count_soc[is]] = L2;
                                    this->non_zero_count_soc[is]++;
                                }
                            }
                        }
                        is++;
                    }
                }
            }
        }
    }
    // 2016-07-19 end, LiuXh

    return;
}


void Atom_pseudo::setup_nonlocal(const ModuleBase::Logger& logger, const bool lspinorb, const int& nspin, MPI_Comm comm)
{
    // get the number of non-local projectors
    int n_projectors = this->nbeta;

    // set the nonlocal projector objects
    NumericalRadial* tmpBeta_lm_new = new NumericalRadial[n_projectors];

    const int nh = this->nh; // zhengdy-soc

    ModuleBase::ComplexMatrix coefficient_D_nc_in(nh * 2, nh * 2); // zhengdy-soc

    int lmaxkb = -1;
    for (int ibeta = 0; ibeta < this->nbeta; ibeta++)
    {
        lmaxkb = std::max(lmaxkb, this->lll[ibeta]);
    }

    std::vector<int> nzeta_(lmaxkb + 1, 0);

    Soc soc;
    if (this->has_so)
    {
        soc.rot_ylm(lmaxkb);
        soc.fcoef.create(1, this->nh, this->nh);
    }

    int ip1 = 0;
    for (int p1 = 0; p1 < n_projectors; p1++) // nbeta
    {
        const int lnow = this->lll[p1];

        const int l1 = this->lll[p1];
        const double j1 = this->jjj[p1];
        for (int m1 = 0; m1 < 2 * l1 + 1; m1++)
        {
            int ip2 = 0;
            for (int p2 = 0; p2 < n_projectors; p2++)
            {
                const int l2 = this->lll[p2];
                const double j2 = this->jjj[p2];
                for (int m2 = 0; m2 < 2 * l2 + 1; m2++)
                {
                    if (l1 == l2 && std::fabs(j1 - j2) < 1e-7)
                    {
                        for (int is1 = 0; is1 < 2; is1++)
                        {
                            for (int is2 = 0; is2 < 2; is2++)
                            {
                                if (this->has_so)
                                {
                                    soc.set_fcoef(l1, l2, is1, is2, m1, m2, j1, j2, 0, ip1, ip2);

                                    coefficient_D_nc_in(ip1 + nh * is1, ip2 + nh * is2)
                                        = this->dion(p1, p2) * soc.fcoef(0, is1, is2, ip1, ip2);
                                    if (p1 != p2)
                                    {
                                        soc.fcoef(0, is1, is2, ip1, ip2) = std::complex<double>(0.0, 0.0);
                                    }
                                }
                                else
                                {
                                    if (is1 == is2 && m1 == m2)
                                    {
                                        coefficient_D_nc_in(ip1 + nh * is1, ip2 + nh * is2) = this->dion(p1, p2);
                                    }
                                }
                            } // end is2
                        }     // end is1
                    }         // end l1==l2
                    ip2++;
                } // end m2
            }     // end p2
            assert(ip2 == nh);
            ip1++;
        } // end m1

        // only keep the nonzero part.
        int cut_mesh = this->mesh;
        for (int ir = this->mesh - 1; ir >= 0; --ir)
        {
            if (std::abs(this->betar(p1, ir)) > 1.0e-10)
            {
                cut_mesh = ir;
                break;
            }
        }
        if (cut_mesh % 2 == 0)
        {
            ++cut_mesh;
        }

        std::vector<double> beta_r(cut_mesh, 0.0);
        for (int ir = 0; ir < cut_mesh; ++ir)
        {
            beta_r[ir] = this->betar(p1, ir);
        }
        
        // Note: 'it' argument in build is used for logging/debugging, passing 0 or a placeholder if not available.
        // Or we can add 'it' (atom type index) to setup_nonlocal arguments if strictly needed.
        // Looking at NumericalRadial::build, 'type' is used for label.
        // We can pass 0 or maybe add type index to Atom_pseudo?
        // Atom_pseudo doesn't know its type index.
        // Let's assume 0 for now or add it to arguments.
        // SetupNonlocalNew passed 'it'.
        // I'll add 'it' to arguments of setup_nonlocal later if needed, but for now 0.
        tmpBeta_lm_new[p1].build(
            lnow, true, cut_mesh, this->r.data(), beta_r.data(), 1, nzeta_[lnow], this->psd, this->itype, false
        );

        nzeta_[lnow] += 1;
    }

    assert(ip1 == nh);

    // We need to pass 'it' to BetaRadials::build too.
    // Let's update setup_nonlocal signature to include 'it'.
    // For now, I will use 0 and fix it in the next step if I decide to add 'it'.
    // Actually, BetaRadials::build takes 'itype'.
    this->beta_radials.build(tmpBeta_lm_new, n_projectors, this->itype, &logger, comm);

    this->set_d_so(coefficient_D_nc_in, n_projectors, nh, this->has_so, lspinorb, nspin);

    delete[] tmpBeta_lm_new;
}
#ifdef __MPI

void Atom_pseudo::bcast_atom_pseudo(MPI_Comm comm)
{
    // ModuleBase::TITLE("Atom_pseudo", "bcast_atom_pseudo");
    // == pseudo_h ==
    int rank;
    MPI_Comm_rank(comm, &rank);
    // int
    Parallel_Common::bcast_int(lmax, comm);
    Parallel_Common::bcast_int(mesh, comm);
    Parallel_Common::bcast_int(nchi, comm);
    Parallel_Common::bcast_int(nbeta, comm);
    Parallel_Common::bcast_int(nv, comm);
    Parallel_Common::bcast_double(zv, comm);

    // double
    Parallel_Common::bcast_double(etotps, comm);
    Parallel_Common::bcast_double(ecutwfc, comm);
    Parallel_Common::bcast_double(ecutrho, comm);

    // bool
    Parallel_Common::bcast_bool(tvanp, comm);
    Parallel_Common::bcast_bool(nlcc, comm);
    Parallel_Common::bcast_bool(has_so, comm);

    // std::string
    Parallel_Common::bcast_string(psd, comm);
    Parallel_Common::bcast_string(pp_type, comm);
    Parallel_Common::bcast_string(xc_func, comm);

    if (rank != 0)
    {
        jjj = std::vector<double>(nbeta, 0.0);
        els = std::vector<std::string>(nchi, "");
        lchi = std::vector<int>(nchi, 0);
        oc = std::vector<double>(nchi, 0.0);
        jchi = std::vector<double>(nchi, 0.0);
        nn = std::vector<int>(nchi, 0);
    }

    Parallel_Common::bcast_double(jjj.data(), nbeta, comm);
    Parallel_Common::bcast_string(els.data(), nchi, comm);
    Parallel_Common::bcast_int(lchi.data(), nchi, comm);
    Parallel_Common::bcast_double(oc.data(), nchi, comm);
    Parallel_Common::bcast_double(jchi.data(), nchi, comm);
    Parallel_Common::bcast_int(nn.data(), nchi, comm);
    // == end of pseudo_h

    // == pseudo_atom ==
    Parallel_Common::bcast_int(msh, comm);
    Parallel_Common::bcast_double(rcut, comm);
    if (rank != 0)
    {
        assert(mesh != 0);
        r = std::vector<double>(mesh, 0.0);
        rab = std::vector<double>(mesh, 0.0);
        rho_atc = std::vector<double>(mesh, 0.0);
        rho_at = std::vector<double>(mesh, 0.0);
        chi.create(nchi, mesh);
    }

    Parallel_Common::bcast_double(r.data(), mesh, comm);
    Parallel_Common::bcast_double(rab.data(), mesh, comm);
    Parallel_Common::bcast_double(rho_atc.data(), mesh, comm);
    Parallel_Common::bcast_double(rho_at.data(), mesh, comm);
    Parallel_Common::bcast_double(chi.c, nchi * mesh, comm);
    // == end of pseudo_atom ==

    // == pseudo_vl ==
    if (rank != 0)
    {
        vloc_at = std::vector<double>(mesh, 0.0);
    }
    Parallel_Common::bcast_double(vloc_at.data(), mesh, comm);
    // == end of pseudo_vl ==

    // == pseudo ==
    if (nbeta == 0) {
        return;
}

    if (rank != 0)
    {
        lll = std::vector<int>(nbeta, 0);
    }
    Parallel_Common::bcast_int(lll.data(), nbeta, comm);
    Parallel_Common::bcast_int(kkbeta, comm);
    Parallel_Common::bcast_int(nh, comm);

    int nr, nc;
    if (rank == 0)
    {
        nr = betar.nr;
        nc = betar.nc;
    }
    Parallel_Common::bcast_int(nr, comm);
    Parallel_Common::bcast_int(nc, comm);

    if (rank != 0)
    {
        betar.create(nr, nc);
        dion.create(nbeta, nbeta);
    }

    // below two 'bcast_double' lines of codes seem to have bugs,
    // on some computers, the code will stuck here for ever.
    // mohan note 2021-04-28
    Parallel_Common::bcast_double(dion.c, nbeta * nbeta, comm);
    Parallel_Common::bcast_double(betar.c, nr * nc, comm);
    // == end of psesudo_nc ==

    // uspp   liuyu 2023-10-03
    if (tvanp)
    {
        Parallel_Common::bcast_int(nqlc, comm);
        if (rank != 0)
        {
            qfuncl.create(nqlc, nbeta * (nbeta + 1) / 2, mesh);
        }
        const int dim = nqlc * nbeta * (nbeta + 1) / 2 * mesh;
        Parallel_Common::bcast_double(qfuncl.ptr, dim, comm);

        if (rank != 0)
        {
            qqq.create(nbeta, nbeta);
        }
        Parallel_Common::bcast_double(qqq.c, nbeta * nbeta, comm);
    }

    return;
}

#endif