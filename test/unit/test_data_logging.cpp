#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "data_logging.hpp"

TEST_CASE("Test ASCII Data Logging")
{
   std::uniform_real_distribution<double> u;
   std::default_random_engine r;

   double time  = u(r);
   constexpr int size = 10;
   mfem::Vector test_vec(size);
   for (int i = 0; i < size; ++i)
   {
      test_vec(i) = u(r);
   }
   mach::ASCIILogger logger;
   logger.saveState(test_vec, "test_vec", 0, time, 0);

   double read_time;
   mfem::Vector read_vec;
   logger.readState("test_vec", 0, 0, read_time, read_vec);

   REQUIRE(read_time == time);

   int read_size = read_vec.Size();
   REQUIRE(read_size == size);

   for (int i = 0; i < size; ++i)
   {
      REQUIRE(read_vec(i) == test_vec(i));
   }
}

TEST_CASE("Test Binary Data Logging")
{
   std::uniform_real_distribution<double> u;
   std::default_random_engine r;

   double time = u(r);
   constexpr int size = 10;
   mfem::Vector test_vec(size);
   for (int i = 0; i < size; ++i)
   {
      test_vec(i) = u(r);
   }
   mach::BinaryLogger logger;
   logger.saveState(test_vec, "test_vec", 0, time, 0);

   double read_time;
   mfem::Vector read_vec;
   logger.readState("test_vec", 0, 0, read_time, read_vec);

   REQUIRE(read_time == time);

   int read_size = read_vec.Size();
   REQUIRE(read_size == size);

   for (int i = 0; i < size; ++i)
   {
      REQUIRE(read_vec(i) == test_vec(i));
   }
}
