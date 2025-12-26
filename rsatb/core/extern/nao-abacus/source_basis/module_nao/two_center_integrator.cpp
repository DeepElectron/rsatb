#include "source_basis/module_nao/two_center_integrator.h"

#include "source_base/vector3.h"
#include "source_base/ylm.h"
#include "source_base/array_pool.h"
#include "source_base/constants.h"
#include <iostream>

TwoCenterIntegrator::TwoCenterIntegrator():
    is_tabulated_(false),
    is_tabulated_pos_(false),
    op_('\0')
{
}

TwoCenterIntegrator::~TwoCenterIntegrator()
{
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ket,
                                   const char op,
                                   const int nr,
                                   const double cutoff)
{
    op_ = op;
    table_.build(bra, ket, op, nr, cutoff);
    RealGauntTable::instance().build(std::max(bra.lmax(), ket.lmax()));
    is_tabulated_ = true;
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ketp,
                                   const RadialCollection& ketm,
                                   const int nr,
                                   const double cutoff)
{
    op_ = 'S';
    tablep_.build(bra, ketp, op_, nr, cutoff);
    tablem_.build(bra, ketm, op_, nr, cutoff);
    table_.build(bra, bra, op_, nr, cutoff);
    RealGauntTable::instance().build(std::max(bra.lmax(), ketp.lmax())); // ketp's lmax should always larger than ketm's.
    is_tabulated_pos_ = true;
    is_tabulated_ = true;
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ket,
                                   const RadialCollection& ketp,
                                   const RadialCollection& ketm,
                                   const int nr,
                                   const double cutoff)
{
    op_ = 'S';
    tablep_.build(bra, ketp, op_, nr, cutoff);
    tablem_.build(bra, ketm, op_, nr, cutoff);
    table_.build(bra, ket, op_, nr, cutoff);
    RealGauntTable::instance().build(std::max(bra.lmax(), ketp.lmax())); 
    is_tabulated_pos_ = true;
    is_tabulated_ = true;
}

void TwoCenterIntegrator::calculate(const int itype1, 
                                    const int l1, 
                                    const int izeta1, 
                                    const int m1, 
                                    const int itype2,
                                    const int l2,
                                    const int izeta2,
                                    const int m2,
	                                const ModuleBase::Vector3<double>& vR, // R = R2 - R
                                    double* out,
                                    double* grad_out) const
{
#ifdef __DEBUG
    assert( is_tabulated_ );
    assert( out || grad_out );
#endif

    if (out) *out = 0.0;
    if (grad_out) std::fill(grad_out, grad_out + 3, 0.0);

    
    double R = vR.norm();
    
    if (R > table_.rmax())
    {
        return;
    }

    // unit vector along R
    ModuleBase::Vector3<double> uR = (R == 0.0 ? ModuleBase::Vector3<double>(0., 0., 1.) : vR / R);

    // generate all necessary real (solid) spherical harmonics
    const int lmax = l1 + l2;
	std::vector<double> Rl_Y((lmax+1) * (lmax+1));
	ModuleBase::Array_Pool<double> grad_Rl_Y((lmax+1) * (lmax+1), 3);

    // R^l * Y is necessary anyway
    ModuleBase::Ylm::rl_sph_harm(l1 + l2, vR[0], vR[1], vR[2], Rl_Y);
    if (grad_out) {
        ModuleBase::Ylm::grad_rl_sph_harm(l1 + l2, vR[0], vR[1], vR[2], Rl_Y.data(), grad_Rl_Y.get_ptr_2D());
    }

    double tmp[2] = {0.0, 0.0};
    double* S_by_Rl = tmp;
    double* d_S_by_Rl = grad_out ? tmp + 1 : nullptr;

    // the sign is given by i^(l1-l2-l) = (-1)^((l1-l2-l)/2)
    int sign = (l1 - l2 - std::abs(l1 - l2)) % 4 == 0 ? 1 : -1;
    for (int l = std::abs(l1 - l2); l <= l1 + l2; l += 2)
    {
        // look up S/R^l and (d/dR)(S/R^l) (if necessary) from the radial table
        table_.lookup(itype1, l1, izeta1, itype2, l2, izeta2, l, R, S_by_Rl, d_S_by_Rl);

		for (int m = -l; m <= l; ++m)
        {
            double G = RealGauntTable::instance()(l1, l2, l, m1, m2, m);

            if (out)
            {
                *out += sign * G * (*S_by_Rl) * Rl_Y[ylm_index(l, m)];
            }

            if (grad_out)
            {
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] += sign * G * ( (*d_S_by_Rl) * uR[i] * Rl_Y[ylm_index(l, m)]
                                                + (*S_by_Rl) * grad_Rl_Y[ylm_index(l, m)][i] );
                }
            }
        }
        sign = -sign;
    }
}

void TwoCenterIntegrator::calculate(const int itype1, 
                                    const int l1, 
                                    const int izeta1, 
                                    const int m1, 
                                    const int itype2,
                                    const int l2,
                                    const int izeta2,
                                    const int m2,
	                                const ModuleBase::Vector3<double>& vR, // R = R2 - R1
                                    const ModuleBase::Vector3<double>& R2,
                                    double* outS,
                                    double* outRx,
                                    double* outRy,
                                    double* outRz
                                    ) const
{
#ifdef __DEBUG
    assert( is_tabulated_pos_ );
    assert( outS );
#endif

    if (outRx) *outRx = 0.0;
    if (outRy) *outRy = 0.0;
    if (outRz) *outRz = 0.0;

    this->calculate(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, vR, outS);
    if (outRx) this->calculate_pos2c(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, 'x', vR, outRx);
    if (outRy) this->calculate_pos2c(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, 'y', vR, outRy);
    if (outRz) this->calculate_pos2c(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, 'z', vR, outRz);


    if (outRx) *outRx += R2[0] * (*outS);
    if (outRy) *outRy += R2[1] * (*outS);
    if (outRz) *outRz += R2[2] * (*outS);
}

void TwoCenterIntegrator::calculate_pos2c(const int itype1, 
                                        const int l1, 
                                        const int izeta1, 
                                        const int m1, 
                                        const int itype2,
                                        const int l2,
                                        const int izeta2,
                                        const int m2,
                                        const char alpha,
                                        const ModuleBase::Vector3<double>& vR, // R = R2 - R1
                                        double* out
                                        ) const
{
#ifdef __DEBUG
    assert( is_tabulated_pos_ );
    assert( out );
#endif

    if (out) *out = 0.0;

    double R = vR.norm();
    if (R > std::min(tablem_.rmax(), tablep_.rmax()))
    {
        return;
    }

    // unit vector along R
    ModuleBase::Vector3<double> uR = (R == 0.0 ? ModuleBase::Vector3<double>(0., 0., 1.) : vR / R);

    // generate all necessary real (solid) spherical harmonics
    const int lmax = l1 + l2 + 1;
	std::vector<double> Rl_Y((lmax+1) * (lmax+1));

    // R^l * Y is necessary anyway
    ModuleBase::Ylm::rl_sph_harm(lmax, vR[0], vR[1], vR[2], Rl_Y);
    double tmp[1] = {0.0};
    double* S_by_Rl = tmp;
    double pref = std::sqrt(ModuleBase::FOUR_PI / 3.0);

    // add a outer loop for decomposed functions from \vec{r}Y(\hat{r})=\sum GY

    int mr[2], mm, m3;
    if (alpha == 'x'){ // m' = 1
        mr[1] = m2+1;
        mr[0] = m2-1;
        mm = 1;
        pref *= -1; // since the spherical harmonics in abacus differ from the standard one with a (-1)^m.
    }else if (alpha == 'z'){ // m' = 0
        mr[0] = m2;
        mr[1] = m2;
        mm = 0;
    }else if (alpha == 'y'){ // m' = -1
        mr[1] = -m2+1;
        mr[0] = -m2-1;
        mm = -1;
        pref *= -1;
    }else{
        throw std::runtime_error("alpha can only taking value among x/y/z");
    }

    // the sign is given by i^(l1-l2-l) = (-1)^((l1-l2-l)/2)
    for (int l3 = (l2-1 < 0) ? l2+1 : l2-1; l3<=l2+1; l3+=2){
        for (int im=0; im<=std::abs(mm); ++im){
            m3 = mr[im];
            if(m3 < -l3 || m3 > l3) continue;

            double Gr = RealGauntTable::instance()(l2, l3, 1, m2, m3, mm);
            int sign = (l1 - l3 - std::abs(l1 - l3)) % 4 == 0 ? 1 : -1;
            for (int l = std::abs(l1 - l3); l <= l1 + l3; l += 2)
            {
                // look up S/R^l and (d/dR)(S/R^l) (if necessary) from the radial table
                if (l3 == l2-1)
                {
                    tablem_.lookup(itype1, l1, izeta1, itype2, l3, izeta2, l, R, S_by_Rl, nullptr);
                } else if (l3 == l2+1)
                {
                    tablep_.lookup(itype1, l1, izeta1, itype2, l3, izeta2, l, R, S_by_Rl, nullptr);
                }
                for (int m = -l; m <= l; ++m)
                {
                    double G = RealGauntTable::instance()(l1, l3, l, m1, m3, m);
                    if (out)
                    {
                        *out += sign * pref * Gr * G * (*S_by_Rl) * Rl_Y[ylm_index(l, m)];
                    }
                }
                sign = -sign;
            }
        }
    }
}


void TwoCenterIntegrator::snap(const int itype1, 
                               const int l1, 
                               const int izeta1, 
                               const int m1, 
                               const int itype2,
	                           const ModuleBase::Vector3<double>& vR,
                               const bool deriv,
                               std::vector<std::vector<double>>& out) const
{
#ifdef __DEBUG
    assert( is_tabulated_ );
#endif

    out.resize(deriv ? 4 : 1);

    // total number of ket functions (including all m!)
    int num_ket = 0;
    for (int l2 = 0; l2 <= table_.lmax_ket(); ++l2)
    {
        num_ket += (2 * l2 + 1) * table_.nchi_ket(itype2, l2);
    }

    if (num_ket == 0)
    {
        return;
    }

	for(size_t i = 0; i < out.size(); ++i)
	{
		out[i].resize(num_ket);
        std::fill(out[i].begin(), out[i].end(), 0.0);
	}

    int index = 0;
    double tmp[3] = {0.0, 0.0, 0.0};
    for (int l2 = 0; l2 <= table_.lmax_ket(); ++l2)
    {
        for (int izeta2 = 0; izeta2 < table_.nchi_ket(itype2, l2); ++izeta2)
        {
            // NOTE: here the order of m is consistent with the rest of ABACUS
            // i.e., 0, 1, -1, 2, -2, 3, -3, ...
            // whether it should be rearranged to -l, -l+1, ..., l will be studied later
            for (int mm2 = 0; mm2 <= 2*l2; ++mm2)
            {
                int m2 = (mm2 % 2 == 0) ? -mm2 / 2 : (mm2 + 1) / 2;
                calculate(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, vR, &out[0][index], deriv ? tmp : nullptr);

                if (deriv)
                {
                    out[1][index] = tmp[0];
                    out[2][index] = tmp[1];
                    out[3][index] = tmp[2];
                }

                ++index;
            }
        }
    }
}

int TwoCenterIntegrator::ylm_index(const int l, const int m) const
{
    return l * l + (m > 0 ? 2 * m - 1 : -2 * m);
}
