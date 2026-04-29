#include "DetectorConstruction.hh"

#include "ExperimentConfig.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4Ellipsoid.hh"
#include "G4Exception.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
#include "G4VSolid.hh"
#include "G4VisAttributes.hh"

#include <algorithm>

namespace
{

G4Material* BuildNamedOreMaterial(const std::string& name, G4NistManager* nist)
{
  if (auto* existing = G4Material::GetMaterial(name, false)) {
    return existing;
  }

  auto* elC = nist->FindOrBuildElement("C");
  auto* elO = nist->FindOrBuildElement("O");
  auto* elAl = nist->FindOrBuildElement("Al");
  auto* elSi = nist->FindOrBuildElement("Si");
  auto* elNa = nist->FindOrBuildElement("Na");
  auto* elMg = nist->FindOrBuildElement("Mg");
  auto* elCa = nist->FindOrBuildElement("Ca");
  auto* elK = nist->FindOrBuildElement("K");
  auto* elFe = nist->FindOrBuildElement("Fe");
  auto* elCu = nist->FindOrBuildElement("Cu");
  auto* elPb = nist->FindOrBuildElement("Pb");
  auto* elS = nist->FindOrBuildElement("S");

  if (name == "Quartz") {
    auto* material = new G4Material("Quartz", 2.65 * g / cm3, 2);
    material->AddElement(elSi, 1);
    material->AddElement(elO, 2);
    return material;
  }
  if (name == "Calcite") {
    auto* material = new G4Material("Calcite", 2.71 * g / cm3, 3);
    material->AddElement(elCa, 1);
    material->AddElement(elC, 1);
    material->AddElement(elO, 3);
    return material;
  }
  if (name == "Albite") {
    auto* material = new G4Material("Albite", 2.62 * g / cm3, 4);
    material->AddElement(elNa, 1);
    material->AddElement(elAl, 1);
    material->AddElement(elSi, 3);
    material->AddElement(elO, 8);
    return material;
  }
  if (name == "Dolomite") {
    auto* material = new G4Material("Dolomite", 2.85 * g / cm3, 4);
    material->AddElement(elCa, 1);
    material->AddElement(elMg, 1);
    material->AddElement(elC, 2);
    material->AddElement(elO, 6);
    return material;
  }
  if (name == "Orthoclase") {
    auto* material = new G4Material("Orthoclase", 2.56 * g / cm3, 4);
    material->AddElement(elK, 1);
    material->AddElement(elAl, 1);
    material->AddElement(elSi, 3);
    material->AddElement(elO, 8);
    return material;
  }
  if (name == "Hematite") {
    auto* material = new G4Material("Hematite", 5.26 * g / cm3, 2);
    material->AddElement(elFe, 2);
    material->AddElement(elO, 3);
    return material;
  }
  if (name == "Magnetite") {
    auto* material = new G4Material("Magnetite", 5.17 * g / cm3, 2);
    material->AddElement(elFe, 3);
    material->AddElement(elO, 4);
    return material;
  }
  if (name == "Pyrite") {
    auto* material = new G4Material("Pyrite", 5.00 * g / cm3, 2);
    material->AddElement(elFe, 1);
    material->AddElement(elS, 2);
    return material;
  }
  if (name == "Chalcopyrite") {
    auto* material = new G4Material("Chalcopyrite", 4.20 * g / cm3, 3);
    material->AddElement(elCu, 1);
    material->AddElement(elFe, 1);
    material->AddElement(elS, 2);
    return material;
  }
  if (name == "Galena") {
    auto* material = new G4Material("Galena", 7.50 * g / cm3, 2);
    material->AddElement(elPb, 1);
    material->AddElement(elS, 1);
    return material;
  }

  G4Exception("BuildNamedOreMaterial()", "InvalidOreChoice", FatalException,
              ("Unknown ore material: " + name).c_str());
  return nullptr;
}

G4Material* BuildConfiguredOreMaterial(const ExperimentConfig& config,
                                       G4NistManager* nist)
{
  if (config.oreMaterialMode == OreMaterialMode::AirPath) {
    return nist->FindOrBuildMaterial("G4_AIR");
  }

  auto* primary = BuildNamedOreMaterial(config.orePrimaryMaterial, nist);
  if (config.oreMaterialMode == OreMaterialMode::Single ||
      config.oreSecondaryMassFraction <= 0.0) {
    return primary;
  }

  auto* secondary = BuildNamedOreMaterial(config.oreSecondaryMaterial, nist);
  const auto fraction = std::clamp(config.oreSecondaryMassFraction, 0.0, 1.0);
  const auto primaryFraction = 1.0 - fraction;

  const auto mixtureName =
      config.orePrimaryMaterial + "_mix_" + config.oreSecondaryMaterial + "_" +
      std::to_string(static_cast<int>(fraction * 1000.0));
  if (auto* existing = G4Material::GetMaterial(mixtureName, false)) {
    return existing;
  }

  const auto density = primary->GetDensity() * primaryFraction +
                       secondary->GetDensity() * fraction;
  auto* mixture = new G4Material(mixtureName, density, 2);
  mixture->AddMaterial(primary, primaryFraction);
  mixture->AddMaterial(secondary, fraction);
  return mixture;
}

G4VSolid* BuildOreSolid(const std::string& name,
                        OreShape shape,
                        G4double thickness,
                        G4double halfY,
                        G4double halfZ)
{
  if (shape == OreShape::Slab) {
    return new G4Box(name, thickness / 2.0, halfY, halfZ);
  }

  return new G4Ellipsoid(name, thickness / 2.0, halfY, halfZ);
}

}  // namespace

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(), fScoringVolume(nullptr), fSideScoringVolume(nullptr)
{}

DetectorConstruction::~DetectorConstruction() = default;

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  const auto& config = GetExperimentConfig();
  auto nist = G4NistManager::Instance();

  auto worldMat = nist->FindOrBuildMaterial("G4_AIR");
  auto envMat = nist->FindOrBuildMaterial("G4_AIR");
  auto detMat = nist->FindOrBuildMaterial("G4_Si");
  auto* oreMat = BuildConfiguredOreMaterial(config, nist);

  G4cout << "[OreConfig] experiment = " << config.experimentLabel << G4endl;
  G4cout << "[OreConfig] material_mode = "
         << OreMaterialModeToString(config.oreMaterialMode)
         << ", primary = " << config.orePrimaryMaterial
         << ", secondary = " << config.oreSecondaryMaterial
         << ", secondary_mass_fraction = " << config.oreSecondaryMassFraction
         << G4endl;
  G4cout << "[OreConfig] shape = " << OreShapeToString(config.oreShape)
         << ", thickness = " << config.oreThickness_mm << " mm"
         << ", host density = " << oreMat->GetDensity() / (g / cm3)
         << " g/cm3" << G4endl;
  G4cout << "[OreConfig] heterogeneity = "
         << HeterogeneityModeToString(config.heterogeneityMode) << G4endl;
  G4cout << "[OutputConfig] prefix = " << config.outputPrefix << G4endl;
  G4cout << "[DetectorConfig] layout = " << config.detectorLayout
         << ", source_variant = " << config.sourceVariant << G4endl;

  constexpr G4bool checkOverlaps = true;

  const auto worldX = config.worldX_cm * cm;
  const auto worldY = config.worldY_cm * cm;
  const auto worldZ = config.worldZ_cm * cm;

  auto solidWorld = new G4Box("World", worldX / 2.0, worldY / 2.0, worldZ / 2.0);
  auto logicWorld = new G4LogicalVolume(solidWorld, worldMat, "World");
  auto physWorld = new G4PVPlacement(nullptr, G4ThreeVector(), logicWorld,
                                     "World", nullptr, false, 0, checkOverlaps);

  const auto envX = config.envelopeX_cm * cm;
  const auto envY = config.envelopeY_cm * cm;
  const auto envZ = config.envelopeZ_cm * cm;

  auto solidEnv = new G4Box("Envelope", envX / 2.0, envY / 2.0, envZ / 2.0);
  auto logicEnv = new G4LogicalVolume(solidEnv, envMat, "Envelope");

  new G4PVPlacement(nullptr, G4ThreeVector(), logicEnv, "Envelope", logicWorld,
                    false, 0, checkOverlaps);

  const auto oreThickness = config.oreThickness_mm * mm;
  const auto oreHalfY = config.oreHalfY_mm * mm;
  const auto oreHalfZ = config.oreHalfZ_mm * mm;

  auto* solidOre =
      BuildOreSolid("OreBody", config.oreShape, oreThickness, oreHalfY, oreHalfZ);
  auto* logicOre = new G4LogicalVolume(solidOre, oreMat, "OreBody");

  new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), logicOre, "OreBody",
                    logicEnv, false, 0, checkOverlaps);

  if (config.heterogeneityMode == HeterogeneityMode::Inclusion &&
      config.oreMaterialMode != OreMaterialMode::AirPath) {
    auto* inclusionMat = BuildNamedOreMaterial(config.inclusionMaterial, nist);
    auto* solidInclusion = BuildOreSolid("OreInclusion",
                                         config.inclusionShape,
                                         config.inclusionThickness_mm * mm,
                                         config.inclusionRadiusY_mm * mm,
                                         config.inclusionRadiusZ_mm * mm);
    auto* logicInclusion =
        new G4LogicalVolume(solidInclusion, inclusionMat, "OreInclusion");

    new G4PVPlacement(nullptr,
                      G4ThreeVector(0.0,
                                    config.inclusionOffsetY_mm * mm,
                                    config.inclusionOffsetZ_mm * mm),
                      logicInclusion,
                      "OreInclusion",
                      logicOre,
                      false,
                      0,
                      checkOverlaps);

    auto* inclusionVis = new G4VisAttributes(G4Colour(0.85, 0.45, 0.1, 0.8));
    inclusionVis->SetForceSolid(true);
    logicInclusion->SetVisAttributes(inclusionVis);
  }

  const auto detX = config.detectorThickness_mm * mm;
  const auto detY = config.detectorHalfY_mm * mm;
  const auto detZ = config.detectorHalfZ_mm * mm;

  auto* solidDet =
      new G4Box("TransmissionDetector", detX / 2.0, detY, detZ);
  auto* logicDet =
      new G4LogicalVolume(solidDet, detMat, "TransmissionDetector");

  new G4PVPlacement(nullptr, G4ThreeVector(config.detectorX_cm * cm, 0, 0),
                    logicDet, "TransmissionDetector", logicEnv, false, 0,
                    checkOverlaps);

  G4LogicalVolume* logicSideDet = nullptr;
  if (config.detectorLayout == "transmission_plus_side_scatter") {
    auto* solidSideDet = new G4Box("SideScatterDetector",
                                   config.sideDetectorHalfX_mm * mm,
                                   config.sideDetectorThickness_mm * mm / 2.0,
                                   config.sideDetectorHalfZ_mm * mm);
    logicSideDet =
        new G4LogicalVolume(solidSideDet, detMat, "SideScatterDetector");

    new G4PVPlacement(nullptr,
                      G4ThreeVector(0, config.sideDetectorY_cm * cm, 0),
                      logicSideDet,
                      "SideScatterDetector",
                      logicEnv,
                      false,
                      0,
                      checkOverlaps);
  }

  logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());

  auto* envVis = new G4VisAttributes(G4Colour(0.2, 0.2, 1.0, 0.05));
  envVis->SetForceWireframe(true);
  logicEnv->SetVisAttributes(envVis);

  auto* oreVis = new G4VisAttributes(G4Colour(0.2, 0.8, 0.9, 0.5));
  oreVis->SetForceSolid(true);
  logicOre->SetVisAttributes(oreVis);

  auto* detVis = new G4VisAttributes(G4Colour(1.0, 0.1, 0.1, 0.8));
  detVis->SetForceSolid(true);
  logicDet->SetVisAttributes(detVis);

  if (logicSideDet) {
    auto* sideVis = new G4VisAttributes(G4Colour(1.0, 0.7, 0.1, 0.65));
    sideVis->SetForceSolid(true);
    logicSideDet->SetVisAttributes(sideVis);
  }

  fScoringVolume = logicDet;
  fSideScoringVolume = logicSideDet;
  return physWorld;
}

bool DetectorConstruction::IsDetectorVolume(const G4LogicalVolume* volume) const
{
  return volume != nullptr && (volume == fScoringVolume || volume == fSideScoringVolume);
}

std::string DetectorConstruction::DetectorId(const G4LogicalVolume* volume) const
{
  if (volume == fSideScoringVolume) {
    return "side_scatter";
  }
  if (volume == fScoringVolume) {
    return "transmission";
  }
  return "unknown";
}
