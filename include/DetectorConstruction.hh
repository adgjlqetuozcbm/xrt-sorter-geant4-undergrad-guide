#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"

#include <string>

class G4VPhysicalVolume;
class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;

    G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }
    bool IsDetectorVolume(const G4LogicalVolume* volume) const;
    std::string DetectorId(const G4LogicalVolume* volume) const;

  private:
    G4LogicalVolume* fScoringVolume;
    G4LogicalVolume* fSideScoringVolume;
};

#endif
