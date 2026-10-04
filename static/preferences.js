(() => {
  const translations = {
    en: {
      pageTitle: 'Amharic Archive', aboutPageTitle: 'About | Amharic Archive',
      welcomePageTitle: 'Welcome to Amharic Archive | Amharic Archive', welcomeNavRead: 'Read', welcomeNavUnderstand: 'Understand', welcomeNavConnect: 'Connect',
      welcomeOverline: 'SEARCHING SPACE FOR AMHARIC', welcomeLine: 'Welcome to', welcomeLanguage: 'Amharic Archive',
      welcomeDescription: 'Explore, learn and engage with Amharic through structured reading and meaningful content.',
      welcomeGetStarted: 'Get Started', welcomeFooter: 'LANGUAGE · KNOWLEDGE · CONNECTION',
      homeLabel: 'Amharic Archive home', mainNavigation: 'Main navigation',
      brandSubtitle: 'Document intelligence', navWorkbench: 'Workbench', navAbout: 'About',
      languageLabel: 'Language', themeToDark: 'Switch to dark mode', themeToLight: 'Switch to light mode',
      workbenchEyebrow: 'AMHARIC NLP WORKBENCH', amharicHeading: 'AMHARIC', heroTitle: 'Keyword and Topic Extraction.',
      heroDescription: 'Turn an Amharic document into a structured reading: topic hierarchy and relevance-ranked keywords.',
      inputDocument: 'INPUT DOCUMENT', readingPrompt: 'What are you reading?', inputPlaceholder: 'Enter your Amharic text here...',
      inputHint: 'Amharic text only · document-level analysis', analyzeDocument: 'Analyze document', quickStart: 'QUICK START',
      quickStartDescription: 'Paste a paragraph, news item, report, or research excerpt. The system keeps the document intact while extracting its strongest signals.',
      loadSample: 'Load a sample', languageGate: 'LANGUAGE GATE', amharicFirst: 'Amharic-first', outputLabel: 'OUTPUT',
      topicsKeywords: 'Topics + keywords', analysisResult: 'ANALYSIS RESULT', documentSignals: 'Document Archive',
      primaryTopic: 'PRIMARY TOPIC', subtopics: 'SUBTOPICS', relevance: 'RELEVANCE', extractedKeywords: 'EXTRACTED KEYWORDS',
      rankedByRelevance: 'RANKED BY RELEVANCE-', reviewIndependently: 'Review each output independently.', approveAll: 'Approve all',
      saveCorrections: 'Save corrections', correction: 'CORRECTION', reviewPrediction: 'Review prediction', close: 'Close',
      approveCorrection: 'Approve correction', localMemory: 'LOCAL MEMORY', recentAnalyses: 'Recent analyses', refresh: 'Refresh',
      recentEmpty: 'Your recent documents will appear here.', aboutEyebrow: 'SYSTEM GUIDE', aboutTitle: 'Purpose and field guide',
      aboutLead: 'Amharic Archive analyzes an Amharic document and brings its strongest subject signals into view: a primary topic, related subtopics, and keywords ranked by relevance.',
      purposeLabel: 'THE PURPOSE', purposeTitle: 'Make long text easier to scan.',
      purposeDescription: 'The system is a document-level NLP workbench. It helps readers, researchers, and reviewers quickly identify what an Amharic passage is mainly about and which terms carry useful subject meaning. Its output is a starting point for review, not a replacement for reading or judgment.',
      howToUse: 'HOW TO USE IT', stepsTitle: 'From text to reviewed signals', stepEnterTitle: 'Enter a document',
      stepEnterDescription: 'Open Workbench, paste Amharic text into the input area, then choose', stepEnterLimit: '. The current input limit is', characters: 'characters.',
      stepResultsTitle: 'Read the results', stepResultsDescription: "Check the primary topic, supporting subtopics, and ranked keywords. Relevance indicates the model's ranking signal; review the source text before relying on a prediction.",
      stepReviewTitle: 'Review or correct', review: 'Review', stepReviewDescription: 'Use each',
      stepReviewDescriptionEnd: 'control to mark an output right or needing correction. For keywords, click a predicted term to remove or restore it, select other words to add them, or drag across adjacent words to make a phrase.',
      stepApproveTitle: 'Approve, then save', stepApproveDescription: 'marks the remaining predictions as right while preserving items already marked for correction. Choose',
      stepApproveEnd: 'to store your reviewed labels.', resultsFeedbackDetails: 'Results and feedback details', whatYouGet: 'WHAT YOU GET',
      threeViews: 'Three views of the document', primaryTopicTitle: 'Primary topic', primaryTopicDescription: 'The main subject predicted for the document.',
      subtopicsDescription: 'Related themes that add detail to the main subject.', keywords: 'Keywords', keywordsDescription: 'Terms and phrases ranked by their relevance to the text.',
      whatSavingDoes: 'WHAT SAVING DOES', feedbackRetained: 'Feedback is retained for later use',
      savingDescription: 'Saved labels are recorded with the analysis in the dashboard database and can inform feedback-aware future analyses. Saving does not immediately retrain the models or change the current result.',
      goWorkbench: 'workbench', keepContext: 'Keep the document in context.',
      reviewNote: 'Predictions are automated suggestions. Check names, specialized terminology, and the surrounding passage before using the extracted signals in research or publication.',
      looksRight: 'Looks right', needsCorrection: 'Needs correction', reviewAction: 'Review',
      mainTopicCorrection: 'Main topic correction', subtopicCorrection: 'Subtopic correction', keywordCorrection: 'Keyword correction',
      chooseCorrectTopic: 'Choose the correct topic', selectTopic: 'Select a topic', enterAnotherTopic: 'Enter another topic',
      enterTopic: 'Enter the topic', correctSubtopic: 'Correct subtopic', enterCorrectSubtopic: 'Enter the correct subtopic',
      clearSelections: 'Clear new selections', restorePredictions: 'Restore predictions', predictedKeywords: 'Predicted keywords',
      shownRed: 'are shown in red. Click a predicted keyword to mark it for deletion. Click it again to restore it.',
      clickOtherWord: 'Click any other word to select it.', dragPhrase: 'Drag across adjacent words to make one keyword phrase, such as',
      selectPhrases: 'You can select multiple separate phrases.', predicted: 'Predicted:', newSelections: 'New selections:',
      markedDeletion: 'Marked for deletion:', noKeywordsRemain: 'No keywords remain. Keep at least one predicted keyword or select a new keyword.',
      enterCorrection: 'Enter or select a correction before approving.', noDominantTopic: 'No dominant topic returned',
      noSubtopics: 'No secondary topics returned.', noKeywords: 'No keywords returned.', enterText: 'Enter Amharic text to analyze.',
      analysisFailed: 'Analysis failed.', enterAmharic: 'Please enter Amharic language.', reviewBeforeSave: 'Review at least one prediction before saving.',
      textTooLong: 'The document exceeds the maximum input length.', backendUnavailable: 'The NLP backend could not complete this analysis.', analysisNotFound: 'Analysis not found.', feedbackRequired: 'Analysis ID and labeled predictions are required.',
      workbenchTitle: 'Keyword and Topic Extraction.',
      analysisLabel: 'ANALYSIS', keywordAction: 'Select or remove keyword',
      saving: 'Saving...', saved: 'Saved', couldNotSave: 'Could not save labels.', historyUnavailable: 'History is unavailable.',
      deleteAnalysis: 'Delete analysis', confirmDelete: 'Delete this analysis from recent history?', couldNotDelete: 'Could not delete analysis.'
    },
    am: {
      pageTitle: 'የአማርኛ ማህደር', aboutPageTitle: 'ስለ | የአማርኛ ማህደር',
      welcomePageTitle: 'እንኳን ወደ አማርኛ ማህደር በደህና መጡ | የአማርኛ ማህደር', welcomeNavRead: 'አንብብ', welcomeNavUnderstand: 'ተረዳ', welcomeNavConnect: 'ተገናኝ',
      welcomeOverline: 'ለአማርኛ ቃላቶች መፈለጊያ ስፍራ', welcomeLine: 'እንኳን ወደ', welcomeLanguage: 'አማርኛ ማህደር በደህና መጡ',
      welcomeDescription: 'በተዋቀረ ንባብና ትርጉም ባለው ይዘት አማርኛን ይመርምሩ፣ ይማሩ እና ይሳተፉ።',
      welcomeGetStarted: 'ይጀምሩ', welcomeFooter: 'ቋንቋ · እውቀት · ግንኙነት',
      homeLabel: 'የአማርኛ ማህደር መነሻ', mainNavigation: 'ዋና ማሰሻ',
      brandSubtitle: 'የሰነድ መረጃ ትንተና', navWorkbench: 'የሥራ ቦታ', navAbout: 'ስለ',
      languageLabel: 'ቋንቋ', themeToDark: 'ወደ ጨለማ ገጽታ ቀይር', themeToLight: 'ወደ ብርሃን ገጽታ ቀይር',
      workbenchEyebrow: 'የአማርኛ ቋንቋ ቴክኖሎጂ የሥራ ቦታ', amharicHeading: 'አማርኛ', heroTitle: 'ቁልፍ ቃላትና ርዕሶችን ማውጣት',
      heroDescription: 'የአማርኛ ሰነድን ወደ የተዋቀረ ንባብ ይቀይሩ፤ የርዕስ ተዋረድና በተዛማጅነት የተደረደሩ ቁልፍ ቃላት',
      inputDocument: 'የግቤት ሰነድ', readingPrompt: 'ምን እያነበቡ ነው?', inputPlaceholder: 'የአማርኛ ጽሑፍዎን እዚህ ያስገቡ...',
      inputHint: 'የአማርኛ ጽሑፍ ብቻ · በሰነድ ደረጃ ትንተና', analyzeDocument: 'ሰነዱን ተንትን', quickStart: 'ፈጣን ጅምር',
      quickStartDescription: 'አንድ አንቀጽ፣ የዜና ዘገባ፣ ሪፖርት ወይም የምርምር ጽሑፍ ክፍል ያስገቡ። ስርዓቱ የጽሑፉን ይዘት ሳይቀይር ዋና ዋና ቁልፍ መረጃዎችን ያወጣል።',
      loadSample: 'ናሙና ጫን', languageGate: 'የቋንቋ ማጣሪያ', amharicFirst: 'አማርኛ ቅድሚያ', outputLabel: 'ውጤት',
      topicsKeywords: 'ርዕሶች + ቁልፍ ቃላት', analysisResult: 'የትንተና ውጤት', documentSignals: 'የሰነድ ምልክቶች',
      primaryTopic: 'ዋና ርዕስ', subtopics: 'ንዑስ ርዕሶች', relevance: 'ተዛማጅነት', extractedKeywords: 'የወጡ ቁልፍ ቃላት',
      rankedByRelevance: 'በተዛማጅነት የተደረደሩ', reviewIndependently: 'እያንዳንዱን ውጤት ለየብቻ ይገምግሙ።', approveAll: 'ሁሉንም አጽድቅ',
      saveCorrections: 'ማስተካከያዎችን አስቀምጥ', correction: 'ማስተካከያ', reviewPrediction: 'ትንበያውን ገምግም', close: 'ተመለስ',
      approveCorrection: 'ማስተካከያውን አጽድቅ', localMemory: 'የአካባቢ ማህደር', recentAnalyses: 'የቅርብ ጊዜ ትንተናዎች', refresh: 'አድስ',
      recentEmpty: 'የቅርብ ጊዜ ሰነዶችዎ እዚህ ይታያሉ።', aboutEyebrow: 'የስርዓቱ መመሪያ', aboutTitle: 'ዓላማና መመሪያ',
      aboutLead: 'የአማርኛ ማህደር የአማርኛ ሰነድን በመተንተን ዋና የርዕስ ምልክቶቹን ያሳያል፤ ዋና ርዕስ፣ ተዛማጅ ንዑስ ርዕሶችና በተዛማጅነት የተደረደሩ ቁልፍ ቃላት።',
      purposeLabel: 'ዓላማው', purposeTitle: 'ረጅም ጽሑፍን በቀላሉ ለመቃኘት',
      purposeDescription: 'ስርዓቱ በሰነድ ደረጃ የሚሰራ የተፈጥሮ ቋንቋ ሂደት መሳሪያ ነው። አንባቢዎች፣ ተመራማሪዎች እና ገምጋሚዎች የአማርኛ ጽሑፍ በዋናነት ስለምን እንደሚያወራ እና ጠቃሚ የርዕሰ ጉዳይ ትርጉም የሚያስተላልፉትን ቃላት በፍጥነት እንዲለዩ ያግዛል። ስርዓቱ የሚያቀርበው ውጤት ለቀጣይ ግምገማ መነሻ ነው፤ ጽሑፉን በማንበብ የሚደረግን ግንዛቤ ወይም የተመራማሪውን ውሳኔ ለመተካት አይደለም።',
      howToUse: 'እንዴት ይጠቀሙበት', stepsTitle: 'ከጽሑፍ ወደ የተገመገሙ ውጤቶች', stepEnterTitle: 'ሰነድ ያስገቡ',
      stepEnterDescription: 'የሥራ ቦታን ይክፈቱ፣ የአማርኛ ጽሑፍን በጽሁፍ ማስገቢያ ቦታው ያስገቡ፣ ከዚያ', stepEnterLimit: 'የሚለውን ይምረጡ። ማስገባት የሚችሂሉት የጽሁፍ ገደቡ', characters: 'ቃላቶችን ነው።',
      stepResultsTitle: 'ውጤቱን ያንብቡ', stepResultsDescription: 'ዋናውን ርዕሰ ጉዳይ፣ ደጋፊ ንዑስ ርዕሶችን እና በደረጃ የተሰጡ ቁልፍ ቃላትን ይመልከቱ። የአግባብነት መጠኑ (Relevance) የሞዴሉን የደረጃ አሰጣጥ ምልክት ያመለክታል፤ በውጤቱ ላይ ከመመስረትዎ በፊት ዋናውን የምንጭ ጽሑፍ ይገምግሙ።',
      stepReviewTitle: 'ይገምግሙ ወይም ያስተካክሉ', review: 'ገምግም', stepReviewDescription: 'እያንዳንዱን',
      stepReviewDescriptionEnd: 'የሚለውን መቆጣጠሪያ በመጠቀም ውጤቱ ትክክል መሆኑን ወይም ማስተካከያ እንደሚፈልግ ያመልክቱ። ለቁልፍ ቃላት፣ የተገመተውን ቃል በመጫን ሊያስወግዱት ወይም መልሰው ሊያክሉት ይችላሉ፤ ሌሎች ቃላትን በመምረጥ ማከል ወይም ተያያዥ ቃላት ላይ በመጎተት እንደ አንድ ሐረግ ማድረግ ይችላሉ።',
      stepApproveTitle: 'አጽድቀው ያስቀምጡ', stepApproveDescription: 'አስቀድመው ለማስተካከያ ምልክት የተደረገባቸውን በመጠበቅ የቀሩትን ትንበያዎች ትክክል ብሎ ያጸድቃል።',
      stepApproveEnd: 'የተገመገሙ ምልክቶችዎን ለማስቀመጥ ይምረጡ።', resultsFeedbackDetails: 'የውጤትና ግብረመልስ ዝርዝሮች',
      whatYouGet: 'የሚያገኙት', threeViews: 'የሰነዱ ሦስት እይታዎች', primaryTopicTitle: 'ዋና ርዕስ', primaryTopicDescription: 'ሰነዱ በዋናነት የሚያተኩርበት የተገመተ ርዕሰ ጉዳይ።',
      subtopicsDescription: 'ለዋናው ርዕሰ ጉዳይ ተጨማሪ ዝርዝር የሚሰጡ ተዛማጅ ርዕሶች።', keywords: 'ቁልፍ ቃላት', keywordsDescription: 'ከጽሑፉ ጋር ባላቸው አግባብነት ደረጃ የተደረደሩ ቁልፍ ቃላትና ሐረጎች።',
      whatSavingDoes: 'ማስቀመጥ የሚያደርገው', feedbackRetained: 'ግብረመልስ ለወደፊት ይቀመጣል',
      savingDescription: 'የተቀመጡ መለያዎች ከትንተናው ጋር በዳሽቦርዱ የመረጃ ቋት ውስጥ ይመዘገባሉ፤ በዚህም መሠረት የተጠቃሚ ግብረመልስን የሚያስተውሉ የወደፊት ትንተናዎችን ለማሻሻል ሊያገለግሉ ይችላሉ። መረጃ ቋት ውስጥ ማስቀመጥ ግን ሞዴሎቹን ወዲያውኑ እንደገና አያሰለጥንም ወይም ወዲያውኑ ውጤት አይቀይርም።',
      goWorkbench: 'የሥራ ቦታ', keepContext: 'ሰነዱን ከግምት ውስጥ ያስገቡ:',
      reviewNote: 'ውጤቶቹ በራስ-ሰር የተመከሩ ግምቶች ናቸው። ስሞችን፣ ልዩ ቃላትን እና የጽሑፉን አውድ በመመርመር በምርምር ወይም በህትመት ከመጠቀምዎ በፊት ያረጋግጡ።',
      looksRight: 'ትክክል ይመስላል', needsCorrection: 'ማስተካከያ ያስፈልጋል', reviewAction: 'ገምግም',
      mainTopicCorrection: 'ዋና ርዕስ ማስተካከያ', subtopicCorrection: 'ንዑስ ርዕስ ማስተካከያ', keywordCorrection: 'ቁልፍ ቃል ማስተካከያ',
      chooseCorrectTopic: 'ትክክለኛውን ርዕስ ይምረጡ', selectTopic: 'ርዕስ ይምረጡ', enterAnotherTopic: 'ሌላ ርዕስ ያስገቡ',
      enterTopic: 'ርዕሱን ያስገቡ', correctSubtopic: 'ንዑስ ርዕሱን ያስተካክሉ', enterCorrectSubtopic: 'ትክክለኛውን ንዑስ ርዕስ ያስገቡ',
      clearSelections: 'አዲስ ምርጫዎችን አጽዳ', restorePredictions: 'ትንበያዎችን መልስ', predictedKeywords: 'የተገመቱ ቁልፍ ቃላት',
      shownRed: 'በቀይ ይታያሉ። የተገመተውን ቁልፍ ቃል ለመሰረዝ ይጫኑ። እንደገና ለመመለስ እንደገና ይጫኑ።',
      clickOtherWord: 'ሌላ ቃል ለመምረጥ ይጫኑ።', dragPhrase: 'እንደዚህ ያለ አንድ የቁልፍ ቃል ሐረግ ለመፍጠር በተያያዙ ቃላት ላይ ይጎትቱ፤ ለምሳሌ',
      selectPhrases: 'ተለያዩ ሐረጎችንም መምረጥ ይችላሉ።', predicted: 'የተገመቱ:', newSelections: 'አዲስ ምርጫዎች:',
      markedDeletion: 'ለመሰረዝ ምልክት የተደረገባቸው:', noKeywordsRemain: 'ምንም ቁልፍ ቃል አልቀረም። ቢያንስ አንድ የተገመተ ቁልፍ ቃል ይተዉ ወይም አዲስ ይምረጡ።',
      enterCorrection: 'ከማጽደቅዎ በፊት ማስተካከያ ያስገቡ ወይም ይምረጡ።', noDominantTopic: 'ዋና ርዕስ አልተመለሰም',
      noSubtopics: 'ምንም ተጨማሪ ርዕሶች አልተመለሱም።', noKeywords: 'ምንም ቁልፍ ቃላት አልተመለሱም።', enterText: 'ለመተንተን የአማርኛ ጽሑፍ ያስገቡ።',
      analysisFailed: 'ትንተናው አልተሳካም።', enterAmharic: 'እባክዎ የአማርኛ ቋንቋ ጽሑፍ ያስገቡ።', reviewBeforeSave: 'ከማስቀመጥዎ በፊት ቢያንስ አንድ ትንበያ ይገምግሙ።',
      textTooLong: 'ሰነዱ ከሚፈቀደው ከፍተኛ የግቤት ርዝመት በላይ ነው።', backendUnavailable: 'የቋንቋ ትንተና ስርዓቱ ይህን ትንተና ማጠናቀቅ አልቻለም።', analysisNotFound: 'ትንተናው አልተገኘም።', feedbackRequired: 'የትንተና መለያና የተሰየሙ ትንበያዎች ያስፈልጋሉ።',
      workbenchTitle: 'ቁልፍ ቃላትና ርዕሶችን ማውጣት',
      analysisLabel: 'ትንተና', keywordAction: 'ቁልፍ ቃል ይምረጡ ወይም ያስወግዱ',
      saving: 'በማስቀመጥ ላይ...', saved: 'ተቀምጧል', couldNotSave: 'መለያዎችን ማስቀመጥ አልተቻለም።', historyUnavailable: 'ታሪኩን ማግኘት አልተቻለም።',
      deleteAnalysis: 'ትንተናውን ሰርዝ', confirmDelete: 'ይህን ትንተና ከቅርብ ጊዜ ታሪክ ይሰርዙ?', couldNotDelete: 'ትንተናውን መሰረዝ አልተቻለም።'
    }
  };

  const languageKey = 'anlp-language';
  const themeKey = 'anlp-theme';
  const storedLanguage = localStorage.getItem(languageKey);
  let language = storedLanguage === 'am' ? 'am' : 'en';
  const storedTheme = localStorage.getItem(themeKey);
  let theme = storedTheme === 'light' || storedTheme === 'dark'
    ? storedTheme
    : (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  function t(key) {
    return translations[language][key] ?? translations.en[key] ?? key;
  }

  function translateApiError(message) {
    const knownMessages = {
      'Enter Amharic text to analyze.': 'enterText',
      'Please enter Amharic language.': 'enterAmharic',
      'The NLP backend could not complete this analysis.': 'backendUnavailable',
      'Analysis not found.': 'analysisNotFound',
      'analysis_id and labeled predictions are required.': 'feedbackRequired'
    };
    const key = knownMessages[message];
    if (key) return t(key);
    if (/^Text exceeds [\d,]+ characters\.$/.test(message)) return t('textTooLong');
    return message;
  }

  function apiErrorTranslationKey(message) {
    const keys = {
      'Enter Amharic text to analyze.': 'enterText',
      'Please enter Amharic language.': 'enterAmharic',
      'The NLP backend could not complete this analysis.': 'backendUnavailable',
      'Analysis not found.': 'analysisNotFound',
      'analysis_id and labeled predictions are required.': 'feedbackRequired'
    };
    if (keys[message]) return keys[message];
    if (/^Text exceeds [\d,]+ characters\.$/.test(message)) return 'textTooLong';
    return '';
  }

  function updateThemeButton() {
    const button = document.querySelector('#theme-toggle');
    if (!button) return;
    const targetKey = theme === 'dark' ? 'themeToLight' : 'themeToDark';
    button.setAttribute('aria-label', t(targetKey));
    button.setAttribute('title', t(targetKey));
    button.setAttribute('aria-pressed', String(theme === 'dark'));
    button.querySelector('span').textContent = theme === 'dark' ? '☀' : '☾';
  }

  function applyLanguage() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-i18n]').forEach(element => {
      element.textContent = t(element.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
      element.setAttribute('placeholder', t(element.dataset.i18nPlaceholder));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
      const value = t(element.dataset.i18nTitle);
      if (element.tagName === 'TITLE') element.textContent = value;
      else element.setAttribute('title', value);
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(element => {
      element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel));
    });
    const languageSelect = document.querySelector('#language-select');
    if (languageSelect) languageSelect.value = language;
    if (document.querySelector('main.workbench-page')) document.title = t('workbenchTitle') + ' | ' + t('pageTitle');
    updateThemeButton();
    window.dispatchEvent(new CustomEvent('anlp:languagechange', { detail: { language } }));
  }

  function setLanguage(nextLanguage) {
    language = nextLanguage === 'am' ? 'am' : 'en';
    localStorage.setItem(languageKey, language);
    applyLanguage();
  }

  function setTheme(nextTheme) {
    theme = nextTheme === 'dark' ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(themeKey, theme);
    updateThemeButton();
  }

  window.t = t;
  window.translateApiError = translateApiError;
  window.apiErrorTranslationKey = apiErrorTranslationKey;
  window.setLanguage = setLanguage;
  document.documentElement.dataset.theme = theme;
  document.querySelector('#language-select')?.addEventListener('change', event => setLanguage(event.target.value));
  document.querySelector('#theme-toggle')?.addEventListener('click', () => setTheme(theme === 'dark' ? 'light' : 'dark'));
  applyLanguage();
})();
