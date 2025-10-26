// Professional Dart code demonstrating inheritance for body measurement calculations.
// Base class defines a common interface for size calculations.
// Subclasses implement specific formulas for breast and butt sizes.
// Assumptions:
// - Breast size: Calculated as cup size based on bust and underbust measurements (in inches).
//   Cup sizes: AA (0"), A (1"), B (2"), C (3"), D (4"), DD (5"), etc.
// - Butt size: Estimated hip circumference category (Small < 36", Medium 36-40", Large >40").
// Measurements are taken as doubles for precision.

import 'dart:io';

abstract class BodyMeasurementCalculator {
  /// Calculates the size based on provided measurements.
  /// Subclasses must implement this method.
  String calculateSize(double measurement1, double measurement2);

  /// Validates measurements (must be positive).
  bool _isValidMeasurement(double measurement) {
    return measurement > 0;
  }
}

class BreastSizeCalculator extends BodyMeasurementCalculator {
  @override
  String calculateSize(double bust, double underbust) {
    if (!_isValidMeasurement(bust) || !_isValidMeasurement(underbust)) {
      return 'Invalid measurements: Both must be positive numbers.';
    }

    double difference = bust - underbust;
    String cupSize;
    switch (difference.floor()) {
      case 0:
        cupSize = 'AA';
        break;
      case 1:
        cupSize = 'A';
        break;
      case 2:
        cupSize = 'B';
        break;
      case 3:
        cupSize = 'C';
        break;
      case 4:
        cupSize = 'D';
        break;
      case 5:
        cupSize = 'DD';
        break;
      default:
        cupSize = '${difference.floor()} (Custom)';
    }

    int bandSize = underbust.round();
    return 'Breast Size: $bandSize$cupSize';
  }
}

class ButtSizeCalculator extends BodyMeasurementCalculator {
  @override
  String calculateSize(double hip, double waist) {
    // Waist is optional but used for ratio check; ignored for basic category.
    if (!_isValidMeasurement(hip)) {
      return 'Invalid hip measurement: Must be a positive number.';
    }

    String category;
    if (hip < 36) {
      category = 'Small';
    } else if (hip <= 40) {
      category = 'Medium';
    } else {
      category = 'Large';
    }

    return 'Butt Size Category: $category (Hip: ${hip.toStringAsFixed(1)}\")';
  }
}

// Additional functions and extensions related to the body measurement code.
// These build upon the existing inheritance structure to provide more comprehensive
// body profiling, including waist-to-hip ratio (WHR) integration, body shape classification,
// and a composite BodyProfile class for holistic analysis.

// Enum for body shape classifications based on common anthropometric standards.
enum BodyShape { hourglass, pear, apple, rectangle, invertedTriangle }

// Extension to ButtSizeCalculator: Adds WHR calculation for health insights.
// WHR = waist / hip; Healthy ranges: Women <0.85, Men <0.90 (generalized here).
extension WaistToHipRatio on ButtSizeCalculator {
  String calculateWHR(double waist, double hip) {
    if (!_isValidMeasurement(waist) || !_isValidMeasurement(hip)) {
      return 'Invalid measurements: Both must be positive numbers.';
    }
    double whr = waist / hip;
    String healthInsight;
    if (whr < 0.85) {
      healthInsight = 'Low risk (Healthy)';
    } else if (whr < 0.95) {
      healthInsight = 'Moderate risk';
    } else {
      healthInsight = 'High risk (Consult professional)';
    }
    return 'Waist-to-Hip Ratio: ${whr.toStringAsFixed(2)} ($healthInsight)';
  }
}

// New subclass for WaistSizeCalculator, inheriting from the base class.
// Categorizes waist size based on health guidelines (e.g., Women <35", Men <40").
class WaistSizeCalculator extends BodyMeasurementCalculator {
  @override
  String calculateSize(double waist, double height) {
    // Height is optional; used for context but not directly in category.
    if (!_isValidMeasurement(waist)) {
      return 'Invalid waist measurement: Must be a positive number.';
    }

    String category;
    if (waist < 35) {
      category = 'Healthy (Small)';
    } else if (waist <= 40) {
      category = 'Borderline (Medium)';
    } else {
      category = 'At Risk (Large)';
    }

    return 'Waist Size Category: $category (Waist: ${waist.toStringAsFixed(1)}\")';
  }
}

// Composite class for full body profile, integrating multiple calculators.
class BodyProfile {
  final double bust;
  final double underbust;
  final double hip;
  final double waist;
  final double height; // In inches, for BMI context if needed.

  BodyProfile({
    required this.bust,
    required this.underbust,
    required this.hip,
    required this.waist,
    required this.height,
  });

  // Integrates breast, butt, and waist calculations.
  Map<String, String> getFullProfile() {
    var breastCalc = BreastSizeCalculator();
    var buttCalc = ButtSizeCalculator();
    var waistCalc = WaistSizeCalculator();

    return {
      'breast': breastCalc.calculateSize(bust, underbust),
      'butt': buttCalc.calculateSize(hip, waist),
      'waist': waistCalc.calculateSize(waist, height),
      'whr': buttCalc.calculateWHR(waist, hip), // Using extension.
      'shape': determineBodyShape(),
    };
  }

  // Determines body shape based on bust, waist, hip ratios.
  // Simple heuristic: Compares differences in measurements.
  String determineBodyShape() {
    double bustDiff = (bust - waist).abs();
    double hipDiff = (hip - waist).abs();
    double shoulderEstimate = bust * 1.05; // Rough estimate; in practice, measure shoulders.

    if ((bustDiff < 2) && (hipDiff < 2)) {
      return 'Rectangle (Balanced)';
    } else if (hipDiff > bustDiff + 2) {
      return 'Pear (Lower body emphasis)';
    } else if (bustDiff > hipDiff + 2) {
      return 'Inverted Triangle (Upper body emphasis)';
    } else if (waist < (bust * 0.75) && waist < (hip * 0.75)) {
      return 'Hourglass (Curvy)';
    } else {
      return 'Apple (Central emphasis)';
    }
  }

  // Additional function: Rough BMI estimate (requires weight, but placeholder).
  // BMI = weight (kg) / (height (m))^2; Here, assumes user provides weight.
  String estimateBMI(double weightKg) {
    double heightM = height * 0.0254; // Convert inches to meters.
    if (heightM <= 0 || weightKg <= 0) {
      return 'Invalid inputs for BMI calculation.';
    }
    double bmi = weightKg / (heightM * heightM);
    String category;
    if (bmi < 18.5) {
      category = 'Underweight';
    } else if (bmi < 25) {
      category = 'Normal';
    } else if (bmi < 30) {
      category = 'Overweight';
    } else {
      category = 'Obese';
    }
    return 'Estimated BMI: ${bmi.toStringAsFixed(1)} ($category)';
  }
}

// Further additional functions and extensions building on the body measurement framework.
// These introduce new calculators for arm and thigh measurements, a health score aggregator,
// clothing size recommendations, and enhanced input handling with unit conversions.

// New subclass for ArmSizeCalculator, categorizing upper arm circumference.
// Based on fitness standards: Small <12", Medium 12-14", Large >14".
class ArmSizeCalculator extends BodyMeasurementCalculator {
  @override
  String calculateSize(double bicep, double tricep) {
    // Uses average of bicep and tricep for overall arm size.
    if (!_isValidMeasurement(bicep) || !_isValidMeasurement(tricep)) {
      return 'Invalid arm measurements: Both must be positive numbers.';
    }
    double avgArm = (bicep + tricep) / 2;

    String category;
    if (avgArm < 12) {
      category = 'Small';
    } else if (avgArm <= 14) {
      category = 'Medium';
    } else {
      category = 'Large';
    }

    return 'Arm Size Category: $category (Avg: ${avgArm.toStringAsFixed(1)}\")';
  }
}

// New subclass for ThighSizeCalculator, categorizing thigh circumference.
// Standards: Small <20", Medium 20-24", Large >24".
class ThighSizeCalculator extends BodyMeasurementCalculator {
  @override
  String calculateSize(double thigh, double calf) {
    // Calf is contextual; thigh is primary.
    if (!_isValidMeasurement(thigh)) {
      return 'Invalid thigh measurement: Must be a positive number.';
    }

    String category;
    if (thigh < 20) {
      category = 'Small';
    } else if (thigh <= 24) {
      category = 'Medium';
    } else {
      category = 'Large';
    }

    return 'Thigh Size Category: $category (Thigh: ${thigh.toStringAsFixed(1)}\")';
  }
}

// Extension for unit conversion utility on BodyProfile.
extension UnitConversion on BodyProfile {
  // Converts all measurements from inches to cm.
  Map<String, double> convertToCm() {
    double cmFactor = 2.54;
    return {
      'bust': bust * cmFactor,
      'underbust': underbust * cmFactor,
      'hip': hip * cmFactor,
      'waist': waist * cmFactor,
      'height': height * cmFactor,
    };
  }

  // Converts from cm to inches if needed.
  Map<String, double> convertToInches(Map<String, double> cmMeasurements) {
    double inchFactor = 1 / 2.54;
    return {
      'bust': cmMeasurements['bust']! * inchFactor,
      'underbust': cmMeasurements['underbust']! * inchFactor,
      'hip': cmMeasurements['hip']! * inchFactor,
      'waist': cmMeasurements['waist']! * inchFactor,
      'height': cmMeasurements['height']! * inchFactor,
    };
  }
}

// Aggregator class for overall health score based on multiple metrics.
// Scores range 0-100; higher is better. Factors in WHR, BMI, and shape balance.
class HealthScoreAggregator {
  static double calculateHealthScore(BodyProfile profile, double weightKg) {
    // Compute metrics directly without string parsing.
    double whr = profile.waist / profile.hip;
    double heightM = profile.height * 0.0254; // Convert inches to meters.
    double bmi = (heightM > 0 && weightKg > 0) ? weightKg / (heightM * heightM) : 30.0;

    // Scoring logic: WHR (0-30 pts), BMI (0-40 pts), Shape balance (0-30 pts).
    double whrScore = (whr < 0.85) ? 30 : (whr < 0.95 ? 20 : 10);
    double bmiScore = (bmi >= 18.5 && bmi < 25) ? 40 : (bmi < 30 ? 30 : 20);
    double shapeScore = _getShapeScore(profile.determineBodyShape());

    return (whrScore + bmiScore + shapeScore).clamp(0, 100);
  }

  static double _getShapeScore(String shape) {
    switch (shape) {
      case 'Hourglass (Curvy)':
      case 'Rectangle (Balanced)':
        return 30;
      case 'Pear (Lower body emphasis)':
      case 'Inverted Triangle (Upper body emphasis)':
        return 25;
      default:
        return 20;
    }
  }
}

// Function for clothing size recommendations based on profile.
// Simplified US women's sizing; bust/waist/hip driven.
String getClothingSizeRecommendation(BodyProfile profile) {
  double bustIn = profile.bust;
  double waistIn = profile.waist;
  double hipIn = profile.hip;

  // Determine sizes for top, bottom, dress.
  String topSize = _getTopSize(bustIn);
  String bottomSize = _getBottomSize(hipIn, waistIn);
  String dressSize = _getDressSize(bustIn, waistIn, hipIn);

  return 'Recommended Sizes - Top: $topSize, Bottom: $bottomSize, Dress: $dressSize';
}

String _getTopSize(double bust) {
  if (bust < 34) return 'XS';
  if (bust < 36) return 'S';
  if (bust < 38) return 'M';
  if (bust < 40) return 'L';
  return 'XL';
}

String _getBottomSize(double hip, double waist) {
  double avg = (hip + waist) / 2;
  if (avg < 36) return '0-2 (XS/S)';
  if (avg < 38) return '4-6 (S/M)';
  if (avg < 40) return '8-10 (M/L)';
  return '12+ (L/XL)';
}

String _getDressSize(double bust, double waist, double hip) {
  double maxMeasurement = [bust, waist, hip].reduce((a, b) => a > b ? a : b);
  if (maxMeasurement < 36) return 'XS';
  if (maxMeasurement < 38) return 'S';
  if (maxMeasurement < 40) return 'M';
  if (maxMeasurement < 42) return 'L';
  return 'XL';
}

// Helper function to get double input from user with prompt and validation.
double _getDoubleInput(String prompt) {
  double? value;
  while (true) {
    stdout.write(prompt);
    String? input = stdin.readLineSync();
    if (input != null) {
      value = double.tryParse(input);
      if (value != null && value > 0) {
        break;
      }
      print('Please enter a valid positive number.');
    } else {
      print('Invalid input. Please try again.');
    }
  }
  return value;
}

// Function to display a simple progress bar for input collection.
void _displayProgress(int current, int total) {
  double progress = current / total;
  int barLength = 30;
  int filled = (progress * barLength).floor();
  String bar = '█' * filled + '░' * (barLength - filled);
  print('\n📊 Input Progress: [$bar] ${ (progress * 100).round() }%');
}

// Enhanced interactive main with menu and improved visuals.
void interactiveMain() {
  // Clear screen if possible
  // print('\x1B[2J\x1B[0;0H'); // ANSI clear, but optional for compatibility

  print('''
  ╔══════════════════════════════════════════════════════════════════════╗
  ║                           💎 BODY ANALYZER 💎                      ║
  ║                    Professional Edition v2.0 - Enhanced UI          ║
  ╚══════════════════════════════════════════════════════════════════════╝
  
  Welcome to the advanced Body Measurement Analyzer! 🌟
  This tool provides personalized insights, health metrics, and style tips.
  All measurements in inches (except weight in kg). Privacy assured - local only.
  
  Choose an option:
  1. 🚀 Interactive Analysis (Enter your measurements)
  2. 🎬 Demo with Sample Data
  3. ❌ Exit
  
  Enter your choice (1-3): 
  ''');
  
  String? choice = stdin.readLineSync();
  switch (choice?.trim()) {
    case '1':
      _runInteractiveAnalysis();
      break;
    case '2':
      _runDemo();
      break;
    default:
      print('👋 Thanks for visiting! Goodbye.');
      return;
  }

  print('\nWould you like to run another analysis? (y/n): ');
  if (stdin.readLineSync()?.toLowerCase() == 'y') {
    interactiveMain();
  } else {
    print('🌈 Thank you for using Body Analyzer! Stay healthy! 💕');
  }
}

void _runInteractiveAnalysis() {
  print('\n🎯 Let\'s get started! Follow the prompts below.');
  print('💡 Tip: Use a flexible tape measure for accuracy.\n');

  List<String> prompts = [
    'Enter bust measurement: ',
    'Enter underbust measurement: ',
    'Enter hip measurement: ',
    'Enter waist measurement: ',
    'Enter height (inches): ',
    'Enter weight (kg): ',
    'Enter bicep measurement: ',
    'Enter tricep measurement: ',
    'Enter thigh measurement: ',
    'Enter calf measurement: '
  ];

  List<double> measurements = [];
  for (int i = 0; i < prompts.length; i++) {
    double val = _getDoubleInput(prompts[i]);
    measurements.add(val);
    _displayProgress(i + 1, prompts.length);
  }

  // Unpack measurements
  double bust = measurements[0];
  double underbust = measurements[1];
  double hip = measurements[2];
  double waist = measurements[3];
  double height = measurements[4];
  double weightKg = measurements[5];
  double bicep = measurements[6];
  double tricep = measurements[7];
  double thigh = measurements[8];
  double calf = measurements[9];

  // Create profile and compute
  var profile = BodyProfile(bust: bust, underbust: underbust, hip: hip, waist: waist, height: height);
  var armCalc = ArmSizeCalculator();
  var thighCalc = ThighSizeCalculator();
  var armResult = armCalc.calculateSize(bicep, tricep);
  var thighResult = thighCalc.calculateSize(thigh, calf);
  double healthScore = HealthScoreAggregator.calculateHealthScore(profile, weightKg);
  String clothingRec = getClothingSizeRecommendation(profile);
  var fullProfile = profile.getFullProfile();
  fullProfile['bmi'] = profile.estimateBMI(weightKg);
  fullProfile['arm'] = armResult;
  fullProfile['thigh'] = thighResult;

  _displayResults(profile, fullProfile, healthScore, clothingRec, measurements);
}

void _runDemo() {
  print('\n🔥 Running demo with sample data...');
  var profile = BodyProfile(bust: 36.0, underbust: 32.0, hip: 38.5, waist: 28.0, height: 65.0);
  var armCalc = ArmSizeCalculator();
  var thighCalc = ThighSizeCalculator();
  var armResult = armCalc.calculateSize(13.0, 12.5);
  var thighResult = thighCalc.calculateSize(22.0, 14.0);
  double healthScore = HealthScoreAggregator.calculateHealthScore(profile, 60.0);
  String clothingRec = getClothingSizeRecommendation(profile);
  var fullProfile = profile.getFullProfile();
  fullProfile['bmi'] = profile.estimateBMI(60.0);
  fullProfile['arm'] = armResult;
  fullProfile['thigh'] = thighResult;

  double bust = 36.0, underbust = 32.0, hip = 38.5, waist = 28.0, height = 65.0;
  List<double> measurements = [bust, underbust, hip, waist, height, 60.0, 13.0, 12.5, 22.0, 14.0];
  _displayResults(profile, fullProfile, healthScore, clothingRec, measurements);
}

void _displayResults(BodyProfile profile, Map<String, String> fullProfile, double healthScore, String clothingRec, List<double> measurements) {
  var cmMap = profile.convertToCm();
  Map<String, double> inchMap = {
    'bust': measurements[0],
    'underbust': measurements[1],
    'hip': measurements[2],
    'waist': measurements[3],
    'height': measurements[4],
  };

  // Core Profile Table
  print('\n\n✨ YOUR CORE PROFILE ✨');
  print('════════════════════════════════════════════════════════════════════════');
  print(' | Metric           | Value                                          | ');
  print(' |──────────────────|─────────────────────────────────────────────────| ');
  fullProfile.forEach((key, value) {
    String paddedKey = key.padRight(16);
    String displayValue = value.length > 45 ? '${value.substring(0, 42)}...' : value.padRight(45);
    print(' | $paddedKey | $displayValue | ');
  });
  print(' |──────────────────|─────────────────────────────────────────────────| \n');

  // Health Insights
  String healthEmoji = healthScore >= 80 ? '🟢 Excellent' : healthScore >= 60 ? '🟡 Good' : '🔴 Needs Attention';
  print('🏥 HEALTH INSIGHTS 🏥');
  print('════════════════════════════════════════════════════════════════════════');
  print(' | Metric        | Value                              | ');
  print(' |───────────────|────────────────────────────────────| ');
  print(' | Health Score  | ${healthScore.toStringAsFixed(0)}/100 $healthEmoji | ');
  print(' |───────────────|────────────────────────────────────| \n');

  // Clothing Recommendations
  print('👗 CLOTHING RECOMMENDATIONS 👗');
  print('════════════════════════════════════════════════════════════════════════');
  print(' $clothingRec');
  print('💡 Pro Tip: Consider brands with inclusive sizing for the best fit!\n');

  // Unit Conversion Table
  print('📏 UNIT CONVERSION (Inches to cm) 📏');
  print('════════════════════════════════════════════════════════════════════════');
  print(' | Measurement    | Inches  | cm       | ');
  print(' |────────────────|─────────|──────────| ');
  ['bust', 'underbust', 'hip', 'waist', 'height'].forEach((key) {
    String paddedKey = key.padRight(14);
    print(' | $paddedKey | ${inchMap[key]!.toStringAsFixed(1).padRight(7)} | ${cmMap[key]!.toStringAsFixed(1).padRight(8)} | ');
  });
  print(' |────────────────|─────────|──────────| \n');

  // Summary
  String bmiCategory = fullProfile['bmi']!.split('(')[1].split(')')[0];
  String shape = fullProfile['shape']!;
  print('📋 EXECUTIVE SUMMARY 📋');
  print('════════════════════════════════════════════════════════════════════════');
  print('• 🌟 Body Shape: $shape');
  print('• ❤️ Health Score: ${healthScore.toStringAsFixed(0)}/100 ($healthEmoji)');
  print('• ⚖️ BMI Category: $bmiCategory');
  print('• 👚 Style Tip: Embrace your unique shape - confidence is key! 💖');
  print('\n⚠️  Disclaimer: This is for entertainment and general guidance. Consult a healthcare professional for medical advice.\n');
}

// Run the enhanced main.
void main() => interactiveMain();
