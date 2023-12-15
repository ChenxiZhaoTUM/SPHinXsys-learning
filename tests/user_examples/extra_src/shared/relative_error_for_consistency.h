#include "sphinxsys.h"
using namespace SPH;

class RelativeError : public LocalDynamics, public GeneralDataDelegateInner, public GeneralDataDelegateContactOnly
{
public:
    RelativeError(ComplexRelation& complex_relation) :
        LocalDynamics(complex_relation.getSPHBody()), 
        GeneralDataDelegateInner(complex_relation.getInnerRelation()), 
        GeneralDataDelegateContactOnly(complex_relation.getContactRelation()),
        pos_(particles_->pos_), mass_(particles_->mass_), rho_(particles_->rho_)
    {
        particles_->registerVariable(error_, "RelativeErrorForConsistency");

        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
            contact_mass_.push_back(&(contact_particles_[k]->mass_));
        }
    }

    virtual ~RelativeError() {};

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Real f_analytical = sin(pos_[index_i][0] * pos_[index_i][0] + pos_[index_i][1] * pos_[index_i][1]);
        Real f_sph = 0;
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            Real f_sph_j = sin(pos_[index_j][0] * pos_[index_j][0] + pos_[index_j][1] * pos_[index_j][1]);
            f_sph += f_sph_j * inner_neighborhood.W_ij_[n] * mass_[index_j] / rho_[index_j];
        }


        for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
        {
            StdLargeVec<Real> &contact_mass_k = *(contact_mass_[k]);
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                size_t index_j = contact_neighborhood.j_[n];
                Real f_sph_j = sin(pos_[index_j][0] * pos_[index_j][0] + pos_[index_j][1] * pos_[index_j][1]);
                f_sph += f_sph_j * contact_neighborhood.W_ij_[n] * mass_[index_j] / rho_[index_j];
            }
        }

        error_[index_i] = abs(f_sph - f_analytical) * abs(f_sph - f_analytical) / abs(f_analytical) / abs(f_analytical) * mass_[index_i] / rho_[index_i];
    }

protected:
    StdLargeVec<Real> error_;
    StdLargeVec<Vecd>& pos_;
    StdLargeVec<Real>& mass_, &rho_;
    StdVec<StdLargeVec<Real> *> contact_mass_, contact_rho_;
};

class RelativeErrorSumSingleBody : public BaseIO
{
public:
    RelativeErrorSumSingleBody(IOEnvironment &io_environment, RealBody &sphbody01, RealBody &sphbody02)
        : BaseIO(io_environment), plt_engine_(), reference_(0.0)
    {
        ComplexRelation sphbody01_complex(sphbody01, { &sphbody02 });
        InteractionDynamics<RelativeError> relative_error_for_sphbody01(sphbody01_complex);
        ReduceDynamics<QuantitySummation<Real>> compute_relative_error_sum_for_sphbody01(sphbody01, "RelativeErrorForConsistency");
        relative_error_for_sphbody01.exec();
        error_sum_ = sqrt(compute_relative_error_sum_for_sphbody01.exec());

        filefullpath_output_ = io_environment_.output_folder_ + "/" + "RelativeErrorForConsistency" + ".dat";
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << "\"run_time\""
                << "   ";
        plt_engine_.writeAQuantityHeader(out_file, reference_, "RelativeErrorForConsistency");
        out_file << "\n";
        out_file.close();
    }

    virtual ~RelativeErrorSumSingleBody() {};

    virtual void writeToFile(size_t iteration_step = 0) override
    {
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   ";
        plt_engine_.writeAQuantity(out_file, error_sum_);
        out_file << "\n";
        out_file.close();
    };

protected:
    Real error_sum_;
    PltEngine plt_engine_;
    std::string filefullpath_output_;
    Real reference_;
};

class RelativeErrorSum : public BaseIO
{
public:
    RelativeErrorSum(IOEnvironment &io_environment, RealBody &sphbody01, RealBody &sphbody02)
        : BaseIO(io_environment), plt_engine_(), reference_(0.0)
    {
        ComplexRelation sphbody01_complex(sphbody01, { &sphbody02 });
        ComplexRelation sphbody02_complex(sphbody02, { &sphbody01 });
        InteractionDynamics<RelativeError> relative_error_for_sphbody01(sphbody01_complex);
        InteractionDynamics<RelativeError> relative_error_for_sphbody02(sphbody02_complex);
        ReduceDynamics<QuantitySummation<Real>> compute_relative_error_sum_for_sphbody01(sphbody01, "RelativeErrorForConsistency");
        ReduceDynamics<QuantitySummation<Real>> compute_relative_error_sum_for_sphbody02(sphbody02, "RelativeErrorForConsistency");
        relative_error_for_sphbody01.exec();
        relative_error_for_sphbody02.exec();
        error_sum_ = sqrt(compute_relative_error_sum_for_sphbody01.exec() + compute_relative_error_sum_for_sphbody02.exec());

        filefullpath_output_ = io_environment_.output_folder_ + "/" + "RelativeErrorForConsistency" + ".dat";
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << "\"run_time\""
                << "   ";
        plt_engine_.writeAQuantityHeader(out_file, reference_, "RelativeErrorForConsistency");
        out_file << "\n";
        out_file.close();
    }

    virtual ~RelativeErrorSum() {};

    virtual void writeToFile(size_t iteration_step = 0) override
    {
        std::ofstream out_file(filefullpath_output_.c_str(), std::ios::app);
        out_file << GlobalStaticVariables::physical_time_ << "   ";
        plt_engine_.writeAQuantity(out_file, error_sum_);
        out_file << "\n";
        out_file.close();
    };

protected:
    Real error_sum_;
    PltEngine plt_engine_;
    std::string filefullpath_output_;
    Real reference_;
};
