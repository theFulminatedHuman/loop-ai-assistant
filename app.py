from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class HospitalVoiceAssistant:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.df = None
        self.model = None
        self.index = None
        self.hospital_vectors = None
        self.bm25_index = None
        self.corpus = []
        self.conversation_context = {}
        
        self.city_aliases = {
            'bangalore': ['bangalore', 'bengaluru', 'banglore', 'blr', 'bangalore rural', 'bangalore urban'],
            'delhi': ['delhi', 'new delhi', 'newdelhi'],
            'mumbai': ['mumbai', 'bombay'],
            'chennai': ['chennai', 'madras'],
            'kolkata': ['kolkata', 'calcutta'],
            'hyderabad': ['hyderabad', 'hyd'],
            'pune': ['pune', 'poona'],
        }
        
        # Common hospital brand names to look for
        self.hospital_brands = [
            'apollo', 'manipal', 'fortis', 'max', 'medanta', 'narayana', 'columbia asia',
            'aster', 'sparsh', 'sakra', 'vikram', 'hosmat', 'st john', 'nimhans',
            'cloudnine', 'motherhood', 'rainbow', 'care', 'yashoda', 'kims'
        ]
        
        self.load_data()
        self.setup_search_system()
    
    def load_data(self):
        try:
            if not os.path.exists(self.csv_file_path):
                logger.error(f"CSV file not found: {self.csv_file_path}")
                self.df = pd.DataFrame(columns=['HOSPITAL NAME', 'Address', 'CITY'])
                return
            
            self.df = pd.read_csv(self.csv_file_path)
            logger.info(f"Original columns: {self.df.columns.tolist()}")
            
            self.df.columns = [col.strip().upper() for col in self.df.columns]
            
            col_mapping = {}
            for col in self.df.columns:
                if 'HOSPITAL' in col and 'NAME' in col:
                    col_mapping[col] = 'HOSPITAL NAME'
                elif col == 'CITY' or 'CITY' in col:
                    col_mapping[col] = 'CITY'
                elif 'ADDRESS' in col:
                    col_mapping[col] = 'ADDRESS'
            
            self.df = self.df.rename(columns=col_mapping)
            self.df = self.df.fillna('')
            
            if 'CITY' in self.df.columns:
                self.df['CITY_NORMALIZED'] = self.df['CITY'].apply(self.normalize_city)
            
            self.df['search_text'] = self.df.apply(
                lambda row: f"{row.get('HOSPITAL NAME', '')} {row.get('CITY', '')} {row.get('ADDRESS', '')}".lower(), 
                axis=1
            )
            self.corpus = self.df['search_text'].tolist()
            
            logger.info(f"Loaded {len(self.df)} hospitals")
            logger.info(f"Sample hospitals: {self.df['HOSPITAL NAME'].head(5).tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.df = pd.DataFrame(columns=['HOSPITAL NAME', 'ADDRESS', 'CITY'])
    
    def normalize_city(self, city: str) -> str:
        city_lower = str(city).lower().strip()
        for standard, aliases in self.city_aliases.items():
            if any(alias in city_lower for alias in aliases):
                return standard
        return city_lower
    
    def get_city_variations(self, city: str) -> List[str]:
        city_lower = city.lower().strip()
        for standard, aliases in self.city_aliases.items():
            if city_lower in aliases or city_lower == standard:
                return aliases + [standard]
        return [city_lower]
    
    def setup_search_system(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            from rank_bm25 import BM25Okapi
            
            logger.info("Setting up RAG system...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.hospital_vectors = self.model.encode(self.corpus, show_progress_bar=True)
            
            dimension = self.hospital_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            import faiss as f
            f.normalize_L2(self.hospital_vectors)
            self.index.add(self.hospital_vectors.astype('float32'))
            
            tokenized_corpus = [self.preprocess_text(doc).split() for doc in self.corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            logger.info("RAG system setup completed")
        except Exception as e:
            logger.warning(f"RAG setup failed: {e}, using simple search")
            self.model = None
    
    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def search_hospitals_by_name_and_city(self, hospital_name: str = None, city: str = None, limit: int = 5) -> List[Dict]:
        """Search hospitals by specific name and/or city"""
        try:
            logger.info(f"Searching: hospital_name='{hospital_name}', city='{city}', limit={limit}")
            
            filtered_df = self.df.copy()
            
            # Filter by city first
            if city:
                city_variations = self.get_city_variations(city)
                mask = filtered_df['CITY'].apply(
                    lambda x: any(var in str(x).lower() for var in city_variations)
                )
                filtered_df = filtered_df[mask]
                logger.info(f"After city filter: {len(filtered_df)} hospitals")
            
            # Filter by hospital name if specified
            if hospital_name:
                name_clean = self.preprocess_text(hospital_name)
                name_words = set(name_clean.split()) - {'hospital', 'hospitals', 'medical', 'centre', 'center'}
                
                def match_score(row_name):
                    row_clean = self.preprocess_text(str(row_name))
                    row_words = set(row_clean.split())
                    
                    # Check for brand name match
                    for brand in self.hospital_brands:
                        if brand in name_clean and brand in row_clean:
                            return 1.0
                    
                    # Check word overlap
                    if not name_words:
                        return 0
                    common = name_words & row_words
                    return len(common) / len(name_words)
                
                filtered_df['match_score'] = filtered_df['HOSPITAL NAME'].apply(match_score)
                filtered_df = filtered_df[filtered_df['match_score'] > 0.3]
                filtered_df = filtered_df.sort_values('match_score', ascending=False)
                logger.info(f"After name filter: {len(filtered_df)} hospitals")
            
            if filtered_df.empty:
                return []
            
            # Return top results
            results = []
            for idx, row in filtered_df.head(limit).iterrows():
                results.append({
                    'name': row.get('HOSPITAL NAME', 'Unknown'),
                    'address': row.get('ADDRESS', 'Address not available'),
                    'city': row.get('CITY', 'Unknown'),
                    'confidence': row.get('match_score', 1.0) if 'match_score' in row else 1.0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def find_hospitals_in_all_cities(self, hospital_name: str) -> Dict[str, List]:
        """Find a hospital brand across all cities"""
        name_clean = self.preprocess_text(hospital_name)
        
        results_by_city = {}
        for idx, row in self.df.iterrows():
            row_name = self.preprocess_text(str(row.get('HOSPITAL NAME', '')))
            
            # Check if hospital name matches
            match = False
            for brand in self.hospital_brands:
                if brand in name_clean and brand in row_name:
                    match = True
                    break
            
            if not match:
                name_words = set(name_clean.split()) - {'hospital', 'hospitals'}
                row_words = set(row_name.split())
                if name_words and len(name_words & row_words) / len(name_words) > 0.5:
                    match = True
            
            if match:
                city = row.get('CITY', 'Unknown')
                if city not in results_by_city:
                    results_by_city[city] = []
                results_by_city[city].append({
                    'name': row.get('HOSPITAL NAME', 'Unknown'),
                    'address': row.get('ADDRESS', 'Address not available'),
                    'city': city
                })
        
        return results_by_city
    
    def process_query(self, user_query: str, session_id: str) -> Dict:
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = {
                'last_query': '',
                'last_results': [],
                'last_city': None,
                'last_hospital_name': None,
                'needs_clarification': False,
                'clarification_type': None
            }
        
        context = self.conversation_context[session_id]
        
        if context['needs_clarification']:
            return self.handle_clarification(user_query, session_id)
        
        intent_info = self.extract_intent(user_query, context)
        logger.info(f"Intent: {intent_info}")
        
        if intent_info['intent'] == 'find_specific_hospital':
            return self.handle_find_specific_hospital(intent_info, session_id)
        elif intent_info['intent'] == 'find_hospitals':
            return self.handle_find_hospitals(intent_info, session_id)
        elif intent_info['intent'] == 'verify_network':
            return self.handle_network_verification(intent_info, session_id)
        elif intent_info['intent'] == 'count_hospitals':
            return self.handle_count_hospitals(intent_info, session_id)
        elif intent_info['intent'] == 'greeting':
            return self.handle_greeting()
        elif intent_info['intent'] == 'followup':
            return self.handle_followup(intent_info, session_id)
        else:
            return self.handle_out_of_scope()
    
    def extract_intent(self, query: str, context: dict) -> Dict:
        query_lower = query.lower().strip()
        query_clean = self.preprocess_text(query)
        
        entities = {
            'city': None,
            'hospital_name': None,
            'count': 3
        }
        
        # Extract city
        city_pattern = r'\b(bangalore|bengaluru|delhi|mumbai|bombay|pune|kolkata|calcutta|chennai|madras|hyderabad|bangalore\s+rural|bangalore\s+urban)\b'
        city_match = re.search(city_pattern, query_lower)
        if city_match:
            entities['city'] = city_match.group(1)
        
        # Extract hospital brand name
        for brand in self.hospital_brands:
            if brand in query_lower:
                entities['hospital_name'] = brand
                break
        
        # Extract count
        count_patterns = [
            (r'(\d+)\s*hospitals?', lambda m: int(m.group(1))),
            (r'(one|two|three|four|five|six|seven|eight|nine|ten)', 
             lambda m: {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}.get(m.group(1), 3)),
        ]
        for pattern, converter in count_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities['count'] = converter(match)
                break
        
        # Determine intent
        intent = 'unknown'
        
        # Greeting
        if re.match(r'^(hello|hi|hey|good\s*(morning|afternoon|evening))\b', query_lower):
            intent = 'greeting'
        
        # Network verification - "is X in my network"
        elif re.search(r'\b(in\s+my\s+network|in\s+network|covered|confirm)\b', query_lower):
            intent = 'verify_network'
            # Extract hospital name for verification
            if entities['hospital_name']:
                pass  # Already extracted
            else:
                # Try to extract from query
                patterns = [
                    r'(?:is|confirm|check)\s+(?:if\s+)?(.+?)\s+(?:in\s+(?:my\s+)?network|covered)',
                    r'(.+?)\s+(?:in\s+(?:my\s+)?network|covered)',
                ]
                for p in patterns:
                    m = re.search(p, query_lower)
                    if m:
                        name = m.group(1).strip()
                        # Clean up
                        name = re.sub(r'^(is|can|you|confirm|if|whether)\s+', '', name)
                        name = re.sub(r'\s+(is|in|bangalore|delhi|mumbai|chennai|hyderabad|pune|kolkata).*$', '', name)
                        if len(name) > 2:
                            entities['hospital_name'] = name
                        break
        
        # Find specific hospital (Apollo, Manipal, etc.)
        elif entities['hospital_name']:
            intent = 'find_specific_hospital'
        
        # How many hospitals
        elif re.search(r'\b(how\s+many|count|number)\b', query_lower):
            intent = 'count_hospitals'
        
        # General find hospitals
        elif re.search(r'\b(find|search|show|list|tell|give|hospitals?|around|near|look\s+for|where)\b', query_lower):
            if entities['hospital_name']:
                intent = 'find_specific_hospital'
            else:
                intent = 'find_hospitals'
        
        # Follow-up
        elif re.search(r'\b(more|another|different|else|i\s+need|i\s+want)\b', query_lower):
            intent = 'followup'
            if context.get('last_city'):
                entities['city'] = context['last_city']
            if context.get('last_hospital_name'):
                entities['hospital_name'] = context['last_hospital_name']
        
        return {
            'intent': intent,
            'entities': entities,
            'original_query': query
        }
    
    def handle_greeting(self) -> Dict:
        return {
            'text': "Hello! I'm Loop AI, your Hospital Network Assistant. I can help you find hospitals in our network or verify if a specific hospital is covered. Try asking me 'Find Apollo hospitals in Bangalore' or 'Is Manipal Hospital in my network?'",
            'type': 'greeting',
            'needs_clarification': False
        }
    
    def handle_find_specific_hospital(self, intent_info: Dict, session_id: str) -> Dict:
        """Handle finding a specific hospital brand"""
        entities = intent_info['entities']
        hospital_name = entities['hospital_name']
        city = entities['city']
        count = entities['count']
        
        context = self.conversation_context[session_id]
        context['last_hospital_name'] = hospital_name
        
        if not city:
            # Search across all cities and ask for clarification
            results_by_city = self.find_hospitals_in_all_cities(hospital_name)
            
            if not results_by_city:
                return {
                    'text': f"I couldn't find any {hospital_name.title()} hospitals in our network. Would you like me to search for a different hospital?",
                    'type': 'no_results',
                    'needs_clarification': False
                }
            
            cities = list(results_by_city.keys())
            total = sum(len(v) for v in results_by_city.values())
            
            if len(cities) == 1:
                # Only one city, show results
                city = cities[0]
                results = results_by_city[city][:count]
                context['last_city'] = city
                context['last_results'] = results
                
                hospital_list = "\n".join([f"{i+1}. {h['name']} - {h['address']}" for i, h in enumerate(results)])
                return {
                    'text': f"I found {len(results)} {hospital_name.title()} hospital(s) in {city}:\n{hospital_list}",
                    'type': 'hospital_list',
                    'results': results,
                    'needs_clarification': False
                }
            else:
                # Multiple cities - ask for clarification
                city_list = ", ".join(cities[:5])
                context['needs_clarification'] = True
                context['clarification_type'] = 'city_for_hospital'
                context['pending_hospital'] = hospital_name
                context['pending_count'] = count
                
                return {
                    'text': f"I found {total} {hospital_name.title()} hospitals across multiple cities: {city_list}. Which city are you looking for {hospital_name.title()} hospital in?",
                    'type': 'clarification',
                    'needs_clarification': True
                }
        else:
            # City specified - search directly
            results = self.search_hospitals_by_name_and_city(hospital_name, city, count)
            
            context['last_city'] = city
            context['last_results'] = results
            
            if not results:
                return {
                    'text': f"I couldn't find any {hospital_name.title()} hospitals in {city.title()} in our network. Would you like me to search in a different city?",
                    'type': 'no_results',
                    'needs_clarification': False
                }
            
            hospital_list = "\n".join([f"{i+1}. {h['name']} - {h['address']}" for i, h in enumerate(results)])
            return {
                'text': f"I found {len(results)} {hospital_name.title()} hospital(s) in {city.title()}:\n{hospital_list}\n\nWould you like more information about any of these?",
                'type': 'hospital_list',
                'results': results,
                'needs_clarification': False
            }
    
    def handle_find_hospitals(self, intent_info: Dict, session_id: str) -> Dict:
        entities = intent_info['entities']
        city = entities['city']
        count = entities['count']
        
        context = self.conversation_context[session_id]
        
        if not city:
            context['needs_clarification'] = True
            context['clarification_type'] = 'city'
            context['pending_count'] = count
            return {
                'text': "I'd be happy to help you find hospitals. Which city are you looking in?",
                'type': 'clarification',
                'needs_clarification': True
            }
        
        results = self.search_hospitals_by_name_and_city(None, city, count)
        
        context['last_city'] = city
        context['last_results'] = results
        
        if not results:
            return {
                'text': f"I couldn't find any hospitals in {city.title()} in our network. Would you like to try a different city?",
                'type': 'no_results',
                'needs_clarification': False
            }
        
        hospital_list = "\n".join([f"{i+1}. {h['name']} - {h['address']}" for i, h in enumerate(results)])
        return {
            'text': f"I found {len(results)} hospitals in {city.title()}:\n{hospital_list}\n\nWould you like more details about any of these hospitals?",
            'type': 'hospital_list',
            'results': results,
            'needs_clarification': False
        }
    
    def handle_network_verification(self, intent_info: Dict, session_id: str) -> Dict:
        entities = intent_info['entities']
        hospital_name = entities['hospital_name']
        city = entities['city']
        
        context = self.conversation_context[session_id]
        
        if not hospital_name:
            context['needs_clarification'] = True
            context['clarification_type'] = 'hospital_name'
            return {
                'text': "I'd be happy to check if a hospital is in your network. Which hospital would you like me to verify?",
                'type': 'clarification',
                'needs_clarification': True
            }
        
        logger.info(f"Verifying network for: {hospital_name}, city: {city}")
        
        # Search for the hospital
        if city:
            results = self.search_hospitals_by_name_and_city(hospital_name, city, 5)
        else:
            results_by_city = self.find_hospitals_in_all_cities(hospital_name)
            results = []
            for city_results in results_by_city.values():
                results.extend(city_results)
            results = results[:5]
        
        if results:
            best = results[0]
            response = f"Yes! {best['name']} in {best['city']} is in your network. The address is {best['address']}."
            
            if len(results) > 1:
                others = "\n".join([f"  - {r['name']} ({r['city']})" for r in results[1:4]])
                response += f"\n\nI also found these related hospitals in our network:\n{others}"
            
            return {
                'text': response,
                'type': 'network_verification',
                'found': True,
                'results': results,
                'needs_clarification': False
            }
        else:
            response = f"I'm sorry, I couldn't find '{hospital_name}' in our network."
            if not city:
                response += " Could you specify the city? That might help me find it."
            
            return {
                'text': response,
                'type': 'network_verification',
                'found': False,
                'needs_clarification': False
            }
    
    def handle_count_hospitals(self, intent_info: Dict, session_id: str) -> Dict:
        entities = intent_info['entities']
        city = entities['city']
        hospital_name = entities['hospital_name']
        
        context = self.conversation_context[session_id]
        
        if not city:
            context['needs_clarification'] = True
            context['clarification_type'] = 'city'
            context['pending_hospital'] = hospital_name
            return {
                'text': "Which city would you like me to check?",
                'type': 'clarification',
                'needs_clarification': True
            }
        
        results = self.search_hospitals_by_name_and_city(hospital_name, city, 100)
        
        context['last_city'] = city
        
        if hospital_name:
            if not results:
                return {
                    'text': f"I couldn't find any {hospital_name.title()} hospitals in {city.title()} in our network.",
                    'type': 'count_result',
                    'needs_clarification': False
                }
            
            hospital_list = "\n".join([f"{i+1}. {h['name']} - {h['address']}" for i, h in enumerate(results[:10])])
            return {
                'text': f"I found {len(results)} {hospital_name.title()} hospital(s) in {city.title()}:\n{hospital_list}",
                'type': 'count_result',
                'needs_clarification': False
            }
        else:
            return {
                'text': f"There are {len(results)} hospitals in our network in {city.title()}.",
                'type': 'count_result',
                'needs_clarification': False
            }
    
    def handle_followup(self, intent_info: Dict, session_id: str) -> Dict:
        context = self.conversation_context[session_id]
        entities = intent_info['entities']
        
        city = entities['city'] or context.get('last_city')
        hospital_name = entities['hospital_name'] or context.get('last_hospital_name')
        count = entities['count']
        
        if not city:
            context['needs_clarification'] = True
            context['clarification_type'] = 'city'
            context['pending_count'] = count
            return {
                'text': "Which city would you like me to search in?",
                'type': 'clarification',
                'needs_clarification': True
            }
        
        if hospital_name:
            intent_info['entities']['city'] = city
            intent_info['entities']['hospital_name'] = hospital_name
            return self.handle_find_specific_hospital(intent_info, session_id)
        else:
            intent_info['entities']['city'] = city
            return self.handle_find_hospitals(intent_info, session_id)
    
    def handle_clarification(self, user_query: str, session_id: str) -> Dict:
        context = self.conversation_context[session_id]
        clarification_type = context['clarification_type']
        
        context['needs_clarification'] = False
        context['clarification_type'] = None
        
        if clarification_type == 'city' or clarification_type == 'city_for_hospital':
            city = self.preprocess_text(user_query)
            count = context.get('pending_count', 3)
            hospital_name = context.get('pending_hospital')
            
            if hospital_name:
                intent_info = {
                    'intent': 'find_specific_hospital',
                    'entities': {'city': city, 'hospital_name': hospital_name, 'count': count},
                    'original_query': user_query
                }
                return self.handle_find_specific_hospital(intent_info, session_id)
            else:
                intent_info = {
                    'intent': 'find_hospitals',
                    'entities': {'city': city, 'count': count, 'hospital_name': None},
                    'original_query': user_query
                }
                return self.handle_find_hospitals(intent_info, session_id)
        
        elif clarification_type == 'hospital_name':
            # Extract hospital name from response
            hospital_name = user_query.strip()
            for brand in self.hospital_brands:
                if brand in user_query.lower():
                    hospital_name = brand
                    break
            
            intent_info = {
                'intent': 'verify_network',
                'entities': {'hospital_name': hospital_name, 'city': context.get('last_city'), 'count': 3},
                'original_query': user_query
            }
            return self.handle_network_verification(intent_info, session_id)
        
        return self.handle_out_of_scope()
    
    def handle_out_of_scope(self) -> Dict:
        return {
            'text': "I'm sorry, I can't help with that. I am forwarding this to a human agent.",
            'type': 'out_of_scope',
            'end_conversation': True,
            'needs_clarification': False
        }

# Initialize
try:
    assistant = HospitalVoiceAssistant("List of GIPSA Hospitals - Sheet1.csv")
    logger.info("Assistant initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    assistant = None

@app.route('/')
def index():
    return render_template('voice_assistant.html')

@app.route('/api/conversation', methods=['POST'])
def handle_conversation():
    try:
        if assistant is None:
            return jsonify({'response': "System initializing...", 'type': 'error'}), 503
        
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        response = assistant.process_query(user_message, session_id)
        
        return jsonify({
            'response': response['text'],
            'type': response['type'],
            'needs_clarification': response.get('needs_clarification', False),
            'end_conversation': response.get('end_conversation', False)
        })
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'response': "Technical error. Please try again.", 'type': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if assistant else 'initializing',
        'hospitals_loaded': len(assistant.df) if assistant else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
