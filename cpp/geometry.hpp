#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include "config.hpp"

class Geometry
{
  private:
    Config c;
    int nrec, nsrc;

    void load();
    void create();

    struct Receivers {
        float* x;
        float* z;
    };

    struct Sources {
        float* x;
        float* z;
    };

    Receivers createReceivers();
    Sources createSources();

  public:
    Receivers rec;
    Sources src;

    void get();
    void save();
};

#endif
