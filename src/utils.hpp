#ifndef MACH_UTILS
#define MACH_UTILS

#include <exception>
#include <iostream>

namespace mach
{

/*!
 * \class MachException
 * \brief Handles (high-level) exceptions in both serial and parallel
 */
class MachException: public std::exception
{
public:
   /*!
   * \brief class constructor
   * \param[in] err_msg - the error message to be printed
   */
   MachException(std::string err_msg) : error_msg(err_msg) {}

   /*!
   * \brief overwrites inherieted member that returns the a c-string
   */
   virtual const char* what() const noexcept
   {
      return error_msg.c_str();
   }

   /*!
   * \brief Use this to print the message; prints only on root for parallel runs
   */
   void print_message()
   {
      // TODO: handle parallel runs!!!
      std::cerr << error_msg << std::endl;
   }

protected:
   std::string error_msg; ///< message printed to std::cerr
};

} // namespace mach

#endif 