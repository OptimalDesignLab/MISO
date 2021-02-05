#ifndef MACH_INTEGRATOR
#define MACH_INTEGRATOR

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"

#include "mach_input.hpp"

namespace mach
{

class MachIntegrator
{
public:
   template <typename T>
   MachIntegrator(T &x) : self_(new model<T>(x))
   { }
   MachIntegrator(const MachIntegrator &x) : self_(x.self_->copy_())
   { }
   MachIntegrator(MachIntegrator&&) noexcept = default;

   MachIntegrator& operator=(const MachIntegrator &x)
   { MachIntegrator tmp(x); *this = std::move(tmp); return *this; }
   MachIntegrator& operator=(MachIntegrator&&) noexcept = default;

   friend void setInput(const MachIntegrator &integ,
                        const std::string &name,
                        const MachInput &input);

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t* copy_() const = 0;
      virtual void setInput_(const std::string &name,
                             const MachInput &input) const = 0;
   };

   template <typename T>
   class model : concept_t
   {
   public:
      model(T &x) : data_(x) { }
      concept_t* copy_() const override { return new model(*this); }
      void setInput_(const std::string &name,
                     const MachInput &input) const override
      { setInput(data_, name, input); }

      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

void setInput(const MachIntegrator &integ,
              const std::string &name,
              const MachInput &input);

void setInput(const mfem::NonlinearFormIntegrator &integ,
              const std::string &name,
              const MachInput &input);

} // namespace mach

#endif
